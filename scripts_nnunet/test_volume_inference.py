"""
Test Volume-Level Inference nnU-Net 2D (Standalone, Non-Distributed)
====================================================================

TUJUAN
------
Memvalidasi bahwa model nnU-Net 2D hasil training menghasilkan segmentasi
yang valid ketika diberi VOLUME 3D utuh (bukan chunk 2D). Script ini
mandiri: tidak butuh RabbitMQ, Redis, Go backend, maupun Docker. Cukup
venv nnU-Net + checkpoint.

Ini langkah validasi titik paling berisiko sebelum me-refactor worker
BrainNav ke strategi volume-level.

INPUT yang didukung
-------------------
1. Folder DICOM (1 file .dcm per slice) -> di-stack jadi volume 3D
2. File NIfTI (.nii.gz) langsung (mis. subjek ADAM dari nnUNet_raw)

OUTPUT
------
- Mask hasil dalam format NIfTI (.nii.gz)
- Overlay PNG per slice yang mengandung prediksi foreground
- Ringkasan statistik (jumlah voxel foreground, slice ber-prediksi)
- Dice (HANYA jika --ground-truth diberikan)

PRASYARAT (jalankan di lingkungan dengan venv nnU-Net + checkpoint)
------------------------------------------------------------------
  conda deactivate ; conda deactivate
  <path>/nnUNet_env/Scripts/Activate.ps1
  pip install pydicom SimpleITK matplotlib   # bila belum ada
  $env:nnUNet_raw="..."; $env:nnUNet_preprocessed="..."; $env:nnUNet_results="..."

CONTOH PEMAKAIAN
----------------
  # Dari folder DICOM (tanpa ground truth -> hanya mask + overlay + statistik)
  python test_volume_inference.py \
      --model-folder "D:/.../nnUnet-Papavero" \
      --dicom-dir "D:/path/ke/folder_dicom_series" \
      --out-dir "./hasil_test"

  # Dari NIfTI subjek ADAM (dengan ground truth -> hitung Dice)
  python test_volume_inference.py \
      --model-folder "D:/.../nnUnet-Papavero" \
      --nifti "D:/.../nnUNet_raw/Dataset501_ADAM/imagesTr/ADAM_10024_0000.nii.gz" \
      --ground-truth "D:/.../nnUNet_raw/Dataset501_ADAM/labelsTr/ADAM_10024.nii.gz" \
      --out-dir "./hasil_test"

CATATAN
-------
- READ-ONLY terhadap dataset & checkpoint. Hanya menulis ke --out-dir.
- Pakai checkpoint_final.pth, ensemble fold sesuai --folds (default 0-4).
- nnU-Net 2D MEMERLUKAN volume 3D agar preprocessing (resample + ZScore
  berbasis statistik volume) berjalan benar. Memberi slice/chunk 2D mentah
  menghasilkan mask kosong (Dice 0). Itu sebabnya script ini mengirim
  volume utuh ke predictor.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print('[ERROR] nibabel belum terpasang. pip install nibabel')
    sys.exit(1)

try:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as e:
    print(f'[ERROR] nnunetv2/torch tidak tersedia: {e}')
    print('       Jalankan di dalam venv nnU-Net.')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Loader DICOM -> volume 3D (numpy) + spacing
# ---------------------------------------------------------------------------
def _resolve_dicom_dir(dicom_dir: str) -> str:
    """Folder DICOM dari CD sering punya subfolder 'DICOM'. Cari folder yang
    benar-benar berisi file DICOM (punya series). Coba path apa adanya dulu,
    lalu subfolder umum, lalu rekursif satu tingkat."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()

    def has_series(p):
        try:
            return len(reader.GetGDCMSeriesIDs(p)) > 0
        except Exception:
            return False

    if has_series(dicom_dir):
        return dicom_dir
    # subfolder umum pada CD DICOM
    for sub in ['DICOM', 'dicom', 'IMAGES', 'images']:
        cand = str(Path(dicom_dir) / sub)
        if Path(cand).is_dir() and has_series(cand):
            print(f'[INFO] Memakai subfolder DICOM: {cand}')
            return cand
    # cari rekursif (maks 2 tingkat) folder pertama yang punya series
    base = Path(dicom_dir)
    for child in base.rglob('*'):
        if child.is_dir() and has_series(str(child)):
            print(f'[INFO] Subfolder DICOM ditemukan: {child}')
            return str(child)
    return dicom_dir  # biar error di pemanggil dgn pesan jelas


# Kata kunci yang menandakan citra angiografi (modalitas yang relevan untuk
# segmentasi aneurysm). nnU-Net dilatih pada TOF, jadi TOF adalah yang
# paling cocok; MRA/angio lain ditandai sebagai kandidat.
_TOF_KEYWORDS = ['tof', 't.o.f', 'time of flight', 'tone']
_ANGIO_KEYWORDS = ['mra', 'angio', 'angiography', 'cow', 'circle of willis', 'venogram', 'mrv']


def classify_sequence(desc: str, scanning_seq: str = '', seq_name: str = '') -> str:
    """Klasifikasi sederhana berbasis metadata teks -> label triase.

    Return salah satu: 'TOF' (paling cocok), 'ANGIO?' (kandidat angiografi
    lain, mungkin perlu model berbeda), atau 'NON-ANGIO' (tidak cocok untuk
    segmentasi aneurysm).
    """
    blob = ' '.join([desc or '', scanning_seq or '', seq_name or '']).lower()
    if any(k in blob for k in _TOF_KEYWORDS):
        return 'TOF'
    if any(k in blob for k in _ANGIO_KEYWORDS):
        return 'ANGIO?'
    return 'NON-ANGIO'


def list_dicom_series(dicom_dir: str):
    """Cetak semua series + triase TOF/angiografi untuk segmentasi aneurysm."""
    try:
        import SimpleITK as sitk
        from pydicom import dcmread
    except ImportError as e:
        print(f'[ERROR] butuh SimpleITK + pydicom: {e}')
        sys.exit(1)

    resolved = _resolve_dicom_dir(dicom_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(resolved)
    if not series_ids:
        print(f'[ERROR] Tidak ada series DICOM di {resolved}')
        sys.exit(1)

    print(f'\n=== {len(series_ids)} SERIES ditemukan di {resolved} ===\n')
    print('%-3s %-9s %-6s %-6s %-7s %s' % ('#', 'TRIASE', 'Modal', 'Slice', 'Contrast', 'Description'))
    print('-' * 100)
    tof_found, angio_found = [], []
    for i, sid in enumerate(series_ids):
        files = reader.GetGDCMSeriesFileNames(resolved, sid)
        desc, modal, scan_seq, seq_name, contrast = '?', '?', '', '', '-'
        try:
            ds = dcmread(files[0], stop_before_pixels=True, force=True)
            desc = str(getattr(ds, 'SeriesDescription', '?'))
            modal = str(getattr(ds, 'Modality', '?'))
            scan_seq = str(getattr(ds, 'ScanningSequence', ''))
            seq_name = str(getattr(ds, 'SequenceName', ''))
            cba = getattr(ds, 'ContrastBolusAgent', None)
            contrast = 'YES' if cba else '-'
        except Exception:
            pass
        triage = classify_sequence(desc, scan_seq, seq_name)
        if triage == 'TOF':
            tof_found.append(i)
        elif triage == 'ANGIO?':
            angio_found.append(i)
        print('%-3d %-9s %-6s %-6d %-7s %s' % (i, triage, modal, len(files), contrast, desc[:44]))

    print('\n' + '-' * 100)
    if tof_found:
        print(f'[OK] Series TOF terdeteksi pada index: {tof_found}')
        print(f'     -> jalankan inference dengan --series-index {tof_found[0]}')
    elif angio_found:
        print(f'[PERHATIAN] Tidak ada TOF, tapi ada kandidat angiografi (index {angio_found}).')
        print('            Model dilatih TOF; hasil pada angiografi lain mungkin tidak optimal.')
        print('            Periksa Description; bila itu CTA/CE-MRA, butuh model berbeda.')
    else:
        print('[STOP] Tidak ada series TOF/angiografi. Modalitas ini TIDAK cocok')
        print('       untuk segmentasi aneurysm (aneurysm hanya terlihat di citra')
        print('       vaskular: TOF-MRA, CTA). Sistem sebaiknya MENOLAK input ini.')
    print('\nPilih series, lalu jalankan: --series-index <#> atau --series-id <UID>.')


def load_dicom_series(dicom_dir: str, series_id: str = None, series_index: int = None):
    """Baca folder DICOM (1 file/slice) -> (volume ZYX float32, spacing [z,y,x]).

    Jika ada >1 series dan tidak dipilih, akan error dengan instruksi.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        print('[ERROR] SimpleITK belum terpasang. pip install SimpleITK')
        sys.exit(1)

    resolved = _resolve_dicom_dir(dicom_dir)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(resolved)
    if not series_ids:
        print(f'[ERROR] Tidak ada series DICOM ditemukan di {resolved}')
        sys.exit(1)

    # Pilih series
    chosen = None
    if series_id:
        if series_id not in series_ids:
            print(f'[ERROR] series-id tidak ditemukan. Jalankan --list-series untuk lihat daftar.')
            sys.exit(1)
        chosen = series_id
    elif series_index is not None:
        if series_index < 0 or series_index >= len(series_ids):
            print(f'[ERROR] series-index di luar jangkauan (0..{len(series_ids)-1}).')
            sys.exit(1)
        chosen = series_ids[series_index]
    elif len(series_ids) == 1:
        chosen = series_ids[0]
    else:
        print(f'[ERROR] {len(series_ids)} series ditemukan. Wajib pilih dengan')
        print('        --series-index <#> atau --series-id <UID>.')
        print('        Jalankan dulu dengan --list-series untuk melihat daftar.')
        sys.exit(1)

    files = reader.GetGDCMSeriesFileNames(resolved, chosen)
    reader.SetFileNames(files)
    img = reader.Execute()  # SimpleITK image

    # SimpleITK: GetArrayFromImage -> (z, y, x); GetSpacing -> (x, y, z)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    sx, sy, sz = img.GetSpacing()
    spacing_zyx = [float(sz), float(sy), float(sx)]
    print(f'[INFO] DICOM series dimuat: shape(ZYX)={vol.shape}, spacing(ZYX)={spacing_zyx}, {len(files)} slice')
    return vol, spacing_zyx


def load_nifti(nifti_path: str):
    """Baca NIfTI -> (volume, spacing). Volume dikembalikan dalam orientasi
    asli file; spacing dari header (3 elemen pertama)."""
    nii = nib.load(nifti_path)
    vol = nii.get_fdata().astype(np.float32)
    zooms = nii.header.get_zooms()[:3]
    # nnU-Net mengharapkan urutan spacing sesuai sumbu array. Kita kirim
    # apa adanya; untuk dataset ADAM (X,Y,Z) -> spacing (sx,sy,sz).
    spacing = [float(z) for z in zooms]
    print(f'[INFO] NIfTI dimuat: shape={vol.shape}, spacing={spacing}')
    return vol, spacing, nii.affine


# ---------------------------------------------------------------------------
# Inferensi volume-level
# ---------------------------------------------------------------------------
def to_nnunet_input(vol: np.ndarray):
    """Ubah volume (apa pun urutan axis-nya) menjadi format nnU-Net 2D:
    (C, Z, Y, X) dengan C=1. nnU-Net 2D memperlakukan axis pertama spasial
    sebagai 'slice'. Kita asumsikan axis paling kecil = slice (Z).

    Return (arr_czyx, axis_z) di mana axis_z adalah indeks axis slice pada
    volume asli, supaya hasil bisa dikembalikan ke orientasi semula.
    """
    axis_z = int(np.argmin(vol.shape))  # axis dengan jumlah elemen paling sedikit
    vol_zyx = np.moveaxis(vol, axis_z, 0)
    arr = vol_zyx[np.newaxis].astype(np.float32)  # (1, Z, Y, X)
    return arr, axis_z, vol_zyx.shape


def run_inference(predictor, vol: np.ndarray, spacing):
    arr, axis_z, zyx_shape = to_nnunet_input(vol)
    # spacing untuk nnU-Net: [sz, sy, sx] mengikuti urutan (Z,Y,X)
    # Ambil spacing pada urutan axis yang sudah dipindah.
    sp = list(spacing)
    sp_z = sp[axis_z]
    sp_rest = [sp[i] for i in range(len(sp)) if i != axis_z]
    spacing_zyx = [sp_z] + sp_rest if len(sp_rest) == 2 else [999.0, 1.0, 1.0]
    props = {'spacing': spacing_zyx}

    mask = predictor.predict_single_npy_array(
        arr, props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    mask = np.asarray(mask)  # (Z, Y, X)
    # Kembalikan ke orientasi volume asli
    mask_orig = np.moveaxis(mask, 0, axis_z)
    return mask_orig.astype(np.uint8), axis_z


# ---------------------------------------------------------------------------
# Output: NIfTI, overlay PNG, statistik, Dice
# ---------------------------------------------------------------------------
def dice_score(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, gt).sum() / denom


def save_overlays(vol: np.ndarray, mask: np.ndarray, axis_z: int, out_dir: Path, max_png: int = 12):
    """Simpan overlay PNG (slice + mask merah) untuk slice yang ada prediksi FG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[WARN] matplotlib tidak ada, lewati overlay PNG. pip install matplotlib')
        return 0

    vol_zyx = np.moveaxis(vol, axis_z, 0)
    mask_zyx = np.moveaxis(mask, axis_z, 0)
    fg_per_slice = (mask_zyx > 0).reshape(mask_zyx.shape[0], -1).sum(axis=1)
    slices_with_fg = np.where(fg_per_slice > 0)[0]

    if len(slices_with_fg) == 0:
        print('[WARN] Tidak ada slice dengan prediksi foreground -> tidak ada overlay.')
        return 0

    # Pilih sampai max_png slice dengan FG terbanyak
    order = slices_with_fg[np.argsort(-fg_per_slice[slices_with_fg])][:max_png]
    overlay_dir = out_dir / 'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for z in sorted(order):
        img = vol_zyx[z]
        msk = mask_zyx[z]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap='gray')
        # overlay merah semi transparan
        red = np.zeros((*msk.shape, 4))
        red[msk > 0] = [1, 0, 0, 0.45]
        ax.imshow(red)
        ax.set_title(f'slice {z} (FG={int(fg_per_slice[z])} px)')
        ax.axis('off')
        fig.savefig(overlay_dir / f'overlay_slice_{z:03d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f'[OK] {len(order)} overlay PNG disimpan di {overlay_dir}')
    return len(order)


def main():
    ap = argparse.ArgumentParser(description='Standalone volume-level nnU-Net 2D inference test')
    ap.add_argument('--model-folder',
                    help='Folder checkpoint (berisi fold_0..4, plans.json, dataset.json). Wajib kecuali --list-series.')
    ap.add_argument('--dicom-dir', help='Folder DICOM series (boleh folder CD yang berisi subfolder DICOM)')
    ap.add_argument('--nifti', help='File NIfTI .nii.gz (alternatif dari --dicom-dir)')
    ap.add_argument('--ground-truth', help='NIfTI ground-truth untuk hitung Dice (opsional)')
    ap.add_argument('--folds', default='0,1,2,3,4', help='Folds ensemble, comma-separated')
    ap.add_argument('--out-dir', default='./hasil_test_volume', help='Folder output')
    ap.add_argument('--checkpoint', default='checkpoint_final.pth')
    ap.add_argument('--list-series', action='store_true',
                    help='Hanya tampilkan daftar series di --dicom-dir lalu keluar (tanpa inference)')
    ap.add_argument('--series-id', help='SeriesInstanceUID lengkap untuk dipilih (untuk DICOM multi-series)')
    ap.add_argument('--series-index', type=int, help='Nomor series dari --list-series (alternatif --series-id)')
    ap.add_argument('--benchmark-runs', type=int, default=1,
                    help='Jumlah pengulangan inferensi untuk mengukur waktu steady-state per volume (warm model). Default 1.')
    args = ap.parse_args()

    # Mode listing series: tidak perlu model
    if args.list_series:
        if not args.dicom_dir:
            print('[ERROR] --list-series butuh --dicom-dir')
            sys.exit(1)
        list_dicom_series(args.dicom_dir)
        return

    if not args.dicom_dir and not args.nifti:
        print('[ERROR] Wajib salah satu: --dicom-dir ATAU --nifti (atau --list-series)')
        sys.exit(1)
    if not args.model_folder:
        print('[ERROR] --model-folder wajib untuk inference.')
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = tuple(int(x) for x in args.folds.split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 64)
    print(' TEST VOLUME-LEVEL INFERENCE nnU-Net 2D (standalone)')
    print('=' * 64)
    print(f' Model folder : {args.model_folder}')
    print(f' Folds        : {folds}')
    print(f' Device       : {device}')
    print('=' * 64)

    # Muat volume
    affine = np.eye(4)
    if args.dicom_dir:
        vol, spacing = load_dicom_series(args.dicom_dir, args.series_id, args.series_index)
    else:
        vol, spacing, affine = load_nifti(args.nifti)

    # Muat predictor
    print(' Loading nnU-Net predictor (ensemble)...')
    _t_load0 = time.perf_counter()
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder, use_folds=folds, checkpoint_name=args.checkpoint,
    )
    _t_load = time.perf_counter() - _t_load0
    print(f' Predictor loaded in {_t_load:.2f} s (one-time, model warm afterwards).')
    print(' Menjalankan inferensi volume...')

    # Benchmark: ukur waktu inferensi per volume (model sudah warm).
    # Run pertama = warmup (mengabaikan overhead inisialisasi CUDA/cuDNN);
    # run berikutnya diukur untuk steady-state.
    n_runs = max(1, args.benchmark_runs)
    timings = []
    mask = None
    axis_z = None
    for i in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        mask, axis_z = run_inference(predictor, vol, spacing)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _dt = time.perf_counter() - _t0
        timings.append(_dt)
        tag = 'warmup' if (i == 0 and n_runs > 1) else 'measured'
        print(f'   run {i+1}/{n_runs} [{tag}]: {_dt:.3f} s')

    print()
    print('=== TIMING INFERENSI PER VOLUME ===')
    print(f' Device                 : {device} ({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')
    print(f' Model load (one-time)  : {_t_load:.2f} s')
    if n_runs > 1:
        steady = timings[1:]  # buang warmup
        print(f' Inference warmup       : {timings[0]:.3f} s')
        print(f' Inference steady-state : mean {np.mean(steady):.3f} s, min {np.min(steady):.3f} s, '
              f'max {np.max(steady):.3f} s (n={len(steady)})')
    else:
        print(f' Inference (1 volume)   : {timings[0]:.3f} s')

    # Statistik
    fg = int((mask > 0).sum())
    total = int(mask.size)
    mask_zyx = np.moveaxis(mask, axis_z, 0)
    n_slices_fg = int((mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1) > 0).sum())
    print()
    print('=== RINGKASAN STATISTIK ===')
    print(f' Volume shape         : {vol.shape}')
    print(f' Foreground voxel     : {fg} dari {total} ({100*fg/total:.4f}%)')
    print(f' Slice ber-prediksi   : {n_slices_fg} dari {mask_zyx.shape[0]}')

    # Simpan mask NIfTI
    mask_path = out_dir / 'prediksi_mask.nii.gz'
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(mask_path))
    print(f' Mask disimpan        : {mask_path}')

    # Overlay PNG
    save_overlays(vol, mask, axis_z, out_dir)

    # Dice (opsional)
    if args.ground_truth:
        gt_raw = nib.load(args.ground_truth).get_fdata().astype(np.uint8)
        if gt_raw.shape == mask.shape:
            print()
            print('=== EVALUASI ===')
            labels = np.unique(gt_raw)
            print(f' Label ground-truth   : {labels.tolist()}')
            # ADAM: label 1 = aneurysm target; label 2 = excluded/treated (BUKAN target).
            # Model dilatih hanya untuk label 1 (lihat dataset.json), jadi Dice
            # dihitung terhadap label == 1 saja, bukan gt > 0.
            if 2 in labels:
                print(' [INFO] Label 2 terdeteksi (ADAM: excluded/treated, BUKAN target).')
                print('        Dice dihitung HANYA terhadap label 1 (aneurysm target).')
            gt_target = (gt_raw == 1).astype(np.uint8)
            print(f' GT label-1 voxel     : {int(gt_target.sum())}')
            d = dice_score(mask, gt_target)
            print(f' Dice (vs label 1)    : {d:.4f}')
            # Dice alternatif terhadap semua FG (label 1 atau 2) untuk konteks
            d_all = dice_score(mask, (gt_raw > 0).astype(np.uint8))
            print(f' Dice (vs semua FG)   : {d_all:.4f}  (referensi, label 1+2)')
            if d < 0.05:
                print(' [CATATAN] Dice rendah. Wajar untuk: (a) hanya 1 fold (bukan')
                print('           ensemble 5-fold), (b) aneurysm target sangat kecil,')
                print('           (c) model mean Dice CV memang ~0.236.')
            else:
                print(' [OK] Prediksi tumpang tindih dengan aneurysm target.')
        else:
            print(f'[WARN] shape mask {mask.shape} != ground-truth {gt_raw.shape}, Dice dilewati.')

    print()
    print('Selesai. Buka folder output untuk melihat mask & overlay.')


if __name__ == '__main__':
    main()
