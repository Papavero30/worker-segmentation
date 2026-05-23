"""
Test Empiris: Chunk 256x256 vs Full-Slice Inference
====================================================

TUJUAN
------
Mengukur secara empiris seberapa besar penurunan Dice ketika nnU-Net 2D
(yang dilatih dengan patch 640x640 pada slice utuh ~560x560) dipakai untuk
inferensi pada chunk 256x256, dibandingkan inferensi pada slice utuh.

Ini menjawab pertanyaan arsitektural: apakah pipeline distributed harus
mengirim chunk 256x256, atau slice utuh 512x512 per worker?

CARA KERJA
----------
Untuk setiap subjek test (ambil N subjek dari labelsTr yang ada foreground):
  1. Ambil slice 2D yang mengandung aneurysm terbanyak (slice paling informatif).
  2. Resize slice ke 512x512 (mereplikasi preprocessing master BrainNav).
  3. JALUR A (full-slice): inferensi nnU-Net pada slice 512x512 utuh.
  4. JALUR B (chunked): pecah jadi 4 chunk 256x256 (overlap 32, stride 224,
     grid 2x2, persis logika SegmentationService.go), inferensi tiap chunk,
     lalu merge pakai Gaussian weighted blending (sigma=32) seperti master.
  5. Hitung Dice kedua jalur terhadap ground truth slice yang sama.
  6. Print tabel perbandingan + rata-rata gap.

PRASYARAT (jalankan di PC-B, di dalam venv nnUNet_env)
-----------------------------------------------------
  conda deactivate ; conda deactivate
  D:/Nabil/worker-segmentation/nnUNet_env/Scripts/Activate.ps1
  $env:nnUNet_raw="D:/Nabil/nnUNet_Workspace/nnUNet_raw"
  $env:nnUNet_preprocessed="D:/Nabil/nnUNet_Workspace/nnUNet_preprocessed"
  $env:nnUNet_results="D:/Nabil/nnUNet_Workspace/nnUNet_results"
  python test_chunk_vs_fullslice.py --num-subjects 8

CATATAN
-------
- Script ini READ-ONLY terhadap dataset & checkpoint. Tidak mengubah apa pun.
- Pakai checkpoint_final.pth 5-fold ensemble (sama dengan deployment plan).
- Resize pakai order=1 (bilinear) untuk image, order=0 (nearest) untuk label,
  meniru perilaku umum preprocessing 2D.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print('[ERROR] nibabel belum terinstall di venv ini. Jalankan di nnUNet_env.')
    sys.exit(1)

try:
    from scipy.ndimage import zoom
except ImportError:
    print('[ERROR] scipy belum terinstall. pip install scipy di venv.')
    sys.exit(1)

try:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as e:
    print(f'[ERROR] nnunetv2/torch tidak tersedia: {e}')
    print('       Pastikan dijalankan di dalam venv nnUNet_env di PC-B.')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Parameter chunking — HARUS sama dengan SegmentationService.go
# ---------------------------------------------------------------------------
TARGET_SIZE = 512        # master resize slice ke 512x512
CHUNK_SIZE = 256         # ukuran chunk
OVERLAP = 32             # overlap antar chunk
STRIDE = CHUNK_SIZE - OVERLAP   # 224
SIGMA = 32.0             # Gaussian blending sigma (master pakai 32)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice biner. pred & gt array {0,1} shape sama."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        # Keduanya kosong = sempurna (tidak ada FG, prediksi juga tidak ada)
        return 1.0
    return 2.0 * inter / denom


def resize_2d(arr: np.ndarray, target: int, is_label: bool) -> np.ndarray:
    """Resize array 2D ke target x target."""
    h, w = arr.shape
    zoom_factors = (target / h, target / w)
    order = 0 if is_label else 1
    out = zoom(arr, zoom_factors, order=order)
    # Pastikan ukuran tepat (zoom kadang meleset 1 px)
    out = out[:target, :target]
    if out.shape != (target, target):
        padded = np.zeros((target, target), dtype=out.dtype)
        padded[:out.shape[0], :out.shape[1]] = out
        out = padded
    return out


def make_gaussian_weight(size: int, sigma: float) -> np.ndarray:
    """Matriks bobot Gaussian centered, persis rumus master:
       w(x,y) = exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))"""
    cx = cy = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def chunk_positions(full: int, chunk: int, stride: int):
    """Replikasi loop chunkSliceWithOverlap di SegmentationService.go:
       for y := 0; y <= height-chunk; y += stride { for x ... }"""
    positions = []
    y = 0
    while y <= full - chunk:
        x = 0
        while x <= full - chunk:
            positions.append((x, y))
            x += stride
        y += stride
    return positions


def predict(predictor, img2d: np.ndarray) -> np.ndarray:
    """Inferensi nnU-Net 2D pada satu array 2D. Return mask uint8 {0,1}.

    PENTING: nnU-Net 2D internal memperlakukan data sebagai volume 3D dengan
    Z=1. predict_single_npy_array butuh shape (C, Z, H, W) = (1, 1, H, W),
    dan spacing 3 elemen [sz, sy, sx] dengan sz besar (axis pseudo-3D).
    Shape (C, H, W) menyebabkan 'axes don't match array'.
    """
    arr = img2d[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, H, W)
    props = {'spacing': [999.0, 1.0, 1.0]}
    mask = predictor.predict_single_npy_array(
        arr, props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    mask = np.asarray(mask)
    # Output bisa (1, H, W) atau (H, W) — squeeze axis pseudo-Z
    mask = np.squeeze(mask)
    return mask.astype(np.uint8)


def pick_best_slice_axis(lab: np.ndarray):
    """Cari (axis, index) slice dengan foreground terbanyak, dengan
    mencoba KETIGA axis. NIfTI ADAM bisa [X,Y,Z], [Z,Y,X], dll —
    jangan asumsikan axis 0 = slice.

    Return (best_axis, best_index, fg_count) atau None bila tidak ada FG.
    """
    best = None
    for ax in range(lab.ndim):
        moved = np.moveaxis(lab, ax, 0)
        fg = (moved > 0).reshape(moved.shape[0], -1).sum(axis=1)
        if fg.max() == 0:
            continue
        idx = int(np.argmax(fg))
        cnt = int(fg[idx])
        if best is None or cnt > best[2]:
            best = (ax, idx, cnt)
    return best


def take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Ambil slice 2D dari volume di axis & index tertentu."""
    return np.moveaxis(vol, axis, 0)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', default=r'D:\Nabil\nnUNet_Workspace')
    ap.add_argument('--dataset', default='Dataset501_ADAM')
    ap.add_argument('--model-folder',
                    default=r'D:\Nabil\nnUNet_Workspace\nnUNet_results\Dataset501_ADAM\nnUNetTrainer__nnUNetPlans__2d')
    ap.add_argument('--folds', default='0,1,2,3,4',
                    help='Comma-separated folds untuk ensemble')
    ap.add_argument('--num-subjects', type=int, default=8,
                    help='Jumlah subjek yang diuji')
    args = ap.parse_args()

    raw_dir = Path(args.workspace) / 'nnUNet_raw' / args.dataset
    images_tr = raw_dir / 'imagesTr'
    labels_tr = raw_dir / 'labelsTr'

    if not labels_tr.exists():
        print(f'[ERROR] {labels_tr} tidak ada. Cek --workspace / --dataset.')
        sys.exit(1)

    folds = tuple(int(x) for x in args.folds.split(','))

    print('=' * 70)
    print(' TEST EMPIRIS: Chunk 256x256 vs Full-Slice 512x512')
    print('=' * 70)
    print(f' Model folder : {args.model_folder}')
    print(f' Folds        : {folds}')
    print(f' Chunk        : {CHUNK_SIZE}x{CHUNK_SIZE}, overlap {OVERLAP}, stride {STRIDE}')
    print(f' Gaussian     : sigma={SIGMA}')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f' Device       : {device}')
    print(' Loading nnU-Net predictor (ensemble)...')

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth',
    )
    print(' Predictor loaded.\n')

    label_files = sorted(labels_tr.glob('*.nii.gz'))[: args.num_subjects]
    gauss_w = make_gaussian_weight(CHUNK_SIZE, SIGMA)
    positions = chunk_positions(TARGET_SIZE, CHUNK_SIZE, STRIDE)
    print(f' Chunk grid   : {len(positions)} chunks @ positions {positions}\n')

    print('%-14s %12s %12s %10s' % ('Subject', 'Dice_Full', 'Dice_Chunk', 'Gap'))
    print('-' * 52)

    results = []
    diag_printed = False
    for lf in label_files:
        subj = lf.name.replace('.nii.gz', '')
        img_f = images_tr / f'{subj}_0000.nii.gz'
        if not img_f.exists():
            print(f'{subj:<14} [skip: image not found]')
            continue

        vol = nib.load(str(img_f)).get_fdata().astype(np.float32)
        lab = nib.load(str(lf)).get_fdata().astype(np.uint8)
        if vol.shape != lab.shape:
            print(f'{subj:<14} [skip: shape mismatch {vol.shape} vs {lab.shape}]')
            continue

        best = pick_best_slice_axis(lab)
        if best is None:
            print(f'{subj:<14} [skip: no foreground in any axis]')
            continue
        best_axis, z, fg_cnt = best

        img2d = take_slice(vol, best_axis, z)
        gt2d = take_slice(lab, best_axis, z)

        img512 = resize_2d(img2d, TARGET_SIZE, is_label=False)
        gt512 = resize_2d(gt2d, TARGET_SIZE, is_label=True)

        # --- JALUR A: full-slice 512x512 ---
        pred_full = predict(predictor, img512)
        d_full = dice_score(pred_full, gt512)

        # Diagnostik 1x untuk subjek pertama: ungkap kenapa Dice mungkin 0
        if not diag_printed:
            diag_printed = True
            print()
            print('  [DIAG] subjek pertama (%s):' % subj)
            print('    vol.shape=%s lab.shape=%s' % (vol.shape, lab.shape))
            print('    best_axis=%d slice_idx=%d fg_voxel_in_slice=%d'
                  % (best_axis, z, fg_cnt))
            print('    img2d range: [%.3f, %.3f]  img512 range: [%.3f, %.3f]'
                  % (img2d.min(), img2d.max(), img512.min(), img512.max()))
            print('    gt2d FG=%d  gt512 FG=%d (after resize)'
                  % (int((gt2d > 0).sum()), int((gt512 > 0).sum())))
            print('    pred_full FG=%d  (total px=%d)'
                  % (int(pred_full.sum()), pred_full.size))
            print('    pred_full unique values: %s'
                  % np.unique(pred_full)[:10])
            print()

        # --- JALUR B: chunked + Gaussian merge (replikasi master) ---
        acc = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float64)
        wsum = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float64)
        for (x, y) in positions:
            chunk = img512[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]
            cmask = predict(predictor, chunk).astype(np.float64)
            acc[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE] += cmask * gauss_w
            wsum[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE] += gauss_w
        merged = np.zeros_like(acc)
        nz = wsum > 0
        merged[nz] = acc[nz] / wsum[nz]
        pred_chunk = (merged >= 0.5).astype(np.uint8)
        d_chunk = dice_score(pred_chunk, gt512)

        gap = d_full - d_chunk
        results.append((subj, d_full, d_chunk, gap))
        print('%-14s %12.4f %12.4f %+10.4f' % (subj, d_full, d_chunk, gap))

    if not results:
        print('\n[ERROR] Tidak ada subjek yang berhasil diuji.')
        sys.exit(1)

    arr = np.array([[r[1], r[2], r[3]] for r in results])
    print('-' * 52)
    print('%-14s %12.4f %12.4f %+10.4f' % (
        'MEAN', arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()))
    print('%-14s %12.4f %12.4f %+10.4f' % (
        'MEDIAN', np.median(arr[:, 0]), np.median(arr[:, 1]), np.median(arr[:, 2])))
    print('=' * 70)

    mean_gap = arr[:, 2].mean()
    mean_full = arr[:, 0].mean()
    rel = (mean_gap / mean_full * 100) if mean_full > 0 else 0.0
    print()
    print('INTERPRETASI:')
    print(f'  Rata-rata Dice full-slice : {mean_full:.4f}')
    print(f'  Rata-rata Dice chunked    : {arr[:, 1].mean():.4f}')
    print(f'  Rata-rata gap (penurunan) : {mean_gap:.4f} ({rel:.1f}% relatif)')
    print()
    if mean_gap < 0.02:
        print('  => Gap KECIL. Chunked inference aman. Pertahankan strategi chunk.')
    elif mean_gap < 0.08:
        print('  => Gap SEDANG. Chunked masih bisa dipakai dgn dokumentasi limitasi.')
    else:
        print('  => Gap BESAR. Pertimbangkan kirim full-slice 512x512 per worker,')
        print('     atau retrain dengan patch 256x256.')
    print()
    print('CATATAN: ini diukur per 1 slice paling informatif per subjek.')
    print('Untuk angka final laporan, pertimbangkan rerata multi-slice.')


if __name__ == '__main__':
    main()
