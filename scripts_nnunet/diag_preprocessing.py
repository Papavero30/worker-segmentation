"""
Diagnostik: Kenapa inferensi per-slice 2D menghasilkan Dice 0,
padahal cross-validation nnU-Net = 0.236?

HIPOTESIS
---------
nnU-Net 2D dilatih dengan pipeline preprocessing penuh: resampling ke
target spacing + ZScore normalization. Cross-validation memberi VOLUME 3D
utuh ke predictor (preprocessing jalan otomatis). Worker BrainNav memberi
SLICE/CHUNK 2D mentah -> normalisasi tidak setara -> model collapse ke
prediksi semua background.

Script ini membandingkan TIGA cara prediksi pada subjek yang sama:
  A. predict_from_files: volume 3D utuh via file (cara cross-validation).
  B. predict_single_npy_array: volume 3D utuh in-memory.
  C. predict_single_npy_array: 1 slice 2D (cara worker BrainNav saat ini).

Kalau A & B bagus tapi C nol -> terkonfirmasi: masalahnya pipeline
preprocessing per-slice, bukan chunk vs full-slice.

READ-ONLY. Jalankan di PC-B dalam venv nnUNet_env dengan env vars nnUNet_*.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as e:
    print(f'[ERROR] dependency tidak tersedia: {e}')
    sys.exit(1)


def dice(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    d = pred.sum() + gt.sum()
    if d == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, gt).sum() / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', default=r'D:\Nabil\nnUNet_Workspace')
    ap.add_argument('--dataset', default='Dataset501_ADAM')
    ap.add_argument('--model-folder',
                    default=r'D:\Nabil\nnUNet_Workspace\nnUNet_results\Dataset501_ADAM\nnUNetTrainer__nnUNetPlans__2d')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--subject', default='ADAM_10024',
                    help='Subjek dengan Dice tinggi di cross-val (default ADAM_10024 = 0.72)')
    args = ap.parse_args()

    raw = Path(args.workspace) / 'nnUNet_raw' / args.dataset
    img_f = raw / 'imagesTr' / f'{args.subject}_0000.nii.gz'
    lab_f = raw / 'labelsTr' / f'{args.subject}.nii.gz'
    if not img_f.exists() or not lab_f.exists():
        print(f'[ERROR] file subjek {args.subject} tidak ditemukan.')
        sys.exit(1)

    folds = tuple(int(x) for x in args.folds.split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 64)
    print(f' DIAGNOSTIK PREPROCESSING - subjek {args.subject}')
    print('=' * 64)
    print(' Loading predictor (ensemble)...')
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        device=device, verbose=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder, use_folds=folds, checkpoint_name='checkpoint_final.pth',
    )
    print(' Predictor loaded.\n')

    lab = nib.load(str(lab_f)).get_fdata().astype(np.uint8)
    img_nii = nib.load(str(img_f))
    vol = img_nii.get_fdata().astype(np.float32)
    print(f' vol.shape={vol.shape}  total FG voxels={int((lab>0).sum())}')
    print()

    # === A. predict_from_files: volume 3D utuh via file ===
    print(' [A] predict_from_files (volume 3D utuh, cara cross-validation)...')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            in_dir = tmp / 'in'
            out_dir = tmp / 'out'
            in_dir.mkdir()
            out_dir.mkdir()
            # nnU-Net butuh nama _0000.nii.gz
            import shutil
            shutil.copy(str(img_f), str(in_dir / f'{args.subject}_0000.nii.gz'))
            predictor.predict_from_files(
                str(in_dir), str(out_dir),
                save_probabilities=False, overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
            pred_a = nib.load(str(out_dir / f'{args.subject}.nii.gz')).get_fdata()
        d_a = dice(pred_a, lab)
        print(f'     -> Dice(A) = {d_a:.4f}  pred FG voxels={int((pred_a>0).sum())}')
    except Exception as e:
        print(f'     [gagal] {e}')
        d_a = None

    # === B. predict_single_npy_array: volume 3D utuh in-memory ===
    # nnU-Net 2D expects (C, Z, H, W). Volume nib (X,Y,Z) -> (1, Z, X, Y)
    print()
    print(' [B] predict_single_npy_array (volume 3D utuh in-memory)...')
    try:
        vol_czxy = np.transpose(vol, (2, 0, 1))[np.newaxis]  # (1, Z, X, Y)
        sp = img_nii.header.get_zooms()[:3]
        props_b = {'spacing': [float(sp[2]), float(sp[0]), float(sp[1])]}
        pred_b = predictor.predict_single_npy_array(
            vol_czxy.astype(np.float32), props_b,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        pred_b = np.asarray(pred_b)
        # kembalikan ke orientasi lab (X,Y,Z) untuk dice
        pred_b_xyz = np.transpose(pred_b, (1, 2, 0))
        d_b = dice(pred_b_xyz, lab)
        print(f'     -> Dice(B) = {d_b:.4f}  pred FG voxels={int((pred_b>0).sum())}')
    except Exception as e:
        print(f'     [gagal] {e}')
        d_b = None

    # === C. predict_single_npy_array: 1 slice 2D (cara worker BrainNav) ===
    print()
    print(' [C] predict_single_npy_array (1 slice 2D, cara worker saat ini)...')
    fg_per_z = (lab > 0).reshape(-1, lab.shape[2]).sum(axis=0)
    z = int(np.argmax(fg_per_z))
    slice_img = vol[:, :, z]
    slice_gt = lab[:, :, z]
    try:
        arr_c = slice_img[np.newaxis, np.newaxis, :, :].astype(np.float32)
        props_c = {'spacing': [999.0, 1.0, 1.0]}
        pred_c = predictor.predict_single_npy_array(
            arr_c, props_c,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        pred_c = np.squeeze(np.asarray(pred_c))
        d_c = dice(pred_c, slice_gt)
        print(f'     -> Dice(C) = {d_c:.4f}  pred FG px={int((pred_c>0).sum())}'
              f'  (slice z={z}, gt FG px={int((slice_gt>0).sum())})')
    except Exception as e:
        print(f'     [gagal] {e}')
        d_c = None

    print()
    print('=' * 64)
    print(' KESIMPULAN')
    print('=' * 64)
    print(f'   A (volume via file)      : {d_a}')
    print(f'   B (volume in-memory)     : {d_b}')
    print(f'   C (single slice 2D)      : {d_c}')
    print()
    if d_a and d_a > 0.1 and (d_c is None or d_c < 0.05):
        print(' >> TERKONFIRMASI: model OK pada volume 3D, GAGAL pada slice 2D.')
        print('    Akar masalah = pipeline preprocessing per-slice, BUKAN chunk.')
        print('    Worker BrainNav harus kirim volume/multi-slice, atau')
        print('    preprocessing manual (resample+ZScore) sebelum predict.')
    elif d_a and d_c and d_c > 0.1:
        print(' >> Slice 2D juga bekerja. Masalah di script test sebelumnya.')
    else:
        print(' >> Pola belum jelas. Perlu investigasi lanjutan.')


if __name__ == '__main__':
    main()
