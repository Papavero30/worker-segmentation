"""
Filter empty-label subjects dari Dataset501_ADAM.

Subject yang mask-nya semua background (0) tidak berguna untuk training
small-foreground segmentation karena bikin model collapse ke "predict all bg".

Action:
  1. Scan semua label di nnUNet_raw\Dataset501_ADAM\labelsTr
  2. Pindah subject empty (image + label) ke folder _excluded\
  3. Update dataset.json (numTraining)
  4. Print summary

Usage:
  python filter_empty_subjects.py               # interactive (akan minta konfirmasi)
  python filter_empty_subjects.py --yes         # auto-confirm

Reversible: Anda bisa pindah balik file dari _excluded\ kalau berubah pikiran.
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default=r'D:\Nabil\nnUNet_Workspace',
                        help='nnUNet workspace root')
    parser.add_argument('--dataset', default='Dataset501_ADAM',
                        help='Dataset folder name')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation')
    args = parser.parse_args()

    raw_dir   = Path(args.workspace) / 'nnUNet_raw' / args.dataset
    images_tr = raw_dir / 'imagesTr'
    labels_tr = raw_dir / 'labelsTr'
    excluded  = raw_dir / '_excluded'
    excluded_images = excluded / 'imagesTr'
    excluded_labels = excluded / 'labelsTr'

    if not labels_tr.exists():
        print(f'[ERROR] labelsTr tidak ada di {labels_tr}')
        sys.exit(1)

    label_files = sorted(labels_tr.glob('*.nii.gz'))
    if not label_files:
        print(f'[ERROR] Tidak ada file label di {labels_tr}')
        sys.exit(1)

    # Scan semua label
    print(f'Scanning {len(label_files)} label files...\n')
    empty_subjects = []
    ok_subjects = []
    for f in label_files:
        mask = nib.load(str(f)).get_fdata()
        fg_voxels = int((mask > 0).sum())
        subj = f.name.replace('.nii.gz', '')  # e.g. ADAM_10001
        if fg_voxels == 0:
            empty_subjects.append(subj)
        else:
            ok_subjects.append((subj, fg_voxels))

    print(f'=== SCAN RESULT ===')
    print(f'Total subjects   : {len(label_files)}')
    print(f'OK subjects      : {len(ok_subjects)}')
    print(f'Empty subjects   : {len(empty_subjects)} ({100*len(empty_subjects)/len(label_files):.1f}%)')
    print()
    print(f'Empty (akan dipindah ke _excluded\\):')
    for s in empty_subjects:
        print(f'  - {s}')
    print()

    if not empty_subjects:
        print('[OK] Tidak ada subject empty. Tidak perlu filter.')
        sys.exit(0)

    if not args.yes:
        ans = input(f'Pindah {len(empty_subjects)} subject ke _excluded\\ ? (y/N): ')
        if ans.strip().lower() != 'y':
            print('Dibatalkan.')
            sys.exit(0)

    # Buat folder excluded
    excluded_images.mkdir(parents=True, exist_ok=True)
    excluded_labels.mkdir(parents=True, exist_ok=True)

    moved = 0
    for subj in empty_subjects:
        # Label: ADAM_10001.nii.gz
        label_src = labels_tr / f'{subj}.nii.gz'
        label_dst = excluded_labels / f'{subj}.nii.gz'
        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))

        # Image: ADAM_10001_0000.nii.gz (nnUNet format)
        image_src = images_tr / f'{subj}_0000.nii.gz'
        image_dst = excluded_images / f'{subj}_0000.nii.gz'
        if image_src.exists():
            shutil.move(str(image_src), str(image_dst))
        moved += 1
        print(f'  Moved: {subj}')

    # Update dataset.json
    dataset_json_path = raw_dir / 'dataset.json'
    if dataset_json_path.exists():
        with open(dataset_json_path) as f:
            data = json.load(f)
        old_count = data.get('numTraining', '?')
        data['numTraining'] = len(ok_subjects)
        with open(dataset_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'\n[OK] dataset.json updated: numTraining {old_count} -> {len(ok_subjects)}')

    print(f'\n=== DONE ===')
    print(f'Moved {moved} empty subjects to: {excluded}')
    print(f'Active dataset sekarang: {len(ok_subjects)} subjects')
    print()
    print('NEXT STEPS:')
    print('1. HAPUS preprocessed lama (karena dataset berubah):')
    print(f'   Remove-Item -Recurse -Force "{Path(args.workspace) / "nnUNet_preprocessed" / args.dataset}"')
    print('2. HAPUS results lama (training restart bersih):')
    print(f'   Remove-Item -Recurse -Force "{Path(args.workspace) / "nnUNet_results" / args.dataset}"')
    print('3. Re-run plan_and_preprocess + training via run_training.ps1')


if __name__ == '__main__':
    main()
