import os
import shutil
import json
import nibabel as nib
import numpy as np
from pathlib import Path

# ============================================================================
# KONFIGURASI PATH
# Sesuaikan path di bawah ini dengan direktori Anda
# ============================================================================
ADAM_RAW_DIR = r"D:\Nabil\ADAM_release_subjs"
NNUNET_RAW_DIR = r"D:\Nabil\nnUNet_Workspace\nnUNet_raw"

# ID Dataset nnUNet (Harus format DatasetXXX_Nama)
DATASET_ID = 501 
DATASET_NAME = f"Dataset{DATASET_ID}_ADAM"

OUT_DIR = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)
OUT_IMAGES_TR = os.path.join(OUT_DIR, "imagesTr")
OUT_LABELS_TR = os.path.join(OUT_DIR, "labelsTr")

# ============================================================================
# FUNGSI UTAMA
# ============================================================================
def setup_directories():
    os.makedirs(OUT_IMAGES_TR, exist_ok=True)
    os.makedirs(OUT_LABELS_TR, exist_ok=True)
    print(f"Direktori output dibuat di: {OUT_DIR}")

def convert_dataset():
    """
    Fungsi ini melakukan loop pada direktori ADAM dan memindahkan/meng-copy
    file ke format nnUNet. nnUNet membutuhkan akhiran _0000.nii.gz untuk image.
    """
    subjects = [d for d in os.listdir(ADAM_RAW_DIR) if os.path.isdir(os.path.join(ADAM_RAW_DIR, d))]
    
    for subject in subjects:
        subj_dir = os.path.join(ADAM_RAW_DIR, subject)
        
        subj_path = Path(subj_dir)
        
        # Cari file secara rekursif, utamakan yang di folder 'pre'
        image_files = list(subj_path.rglob("pre/TOF.nii.gz"))
        if not image_files:
             image_files = list(subj_path.rglob("TOF.nii.gz"))
             
        label_files = list(subj_path.rglob("aneurysms.nii.gz"))
        
        if image_files and label_files:
            image_path = str(image_files[0])
            label_path = str(label_files[0])
            
            # Format nama nnUNet: ADAM_001_0000.nii.gz (Image), ADAM_001.nii.gz (Label)
            out_image_name = f"ADAM_{subject}_0000.nii.gz"
            out_label_name = f"ADAM_{subject}.nii.gz"
            
            # 1. Image (TOF.nii.gz) -> Symlink / Copy
            try:
                os.symlink(image_path, os.path.join(OUT_IMAGES_TR, out_image_name))
                print(f"[{subject}] Image SUCCESS (Symlink): {image_path} -> {out_image_name}")
            except OSError:
                shutil.copy(image_path, os.path.join(OUT_IMAGES_TR, out_image_name))
                print(f"[{subject}] Image SUCCESS (Copy): {image_path} -> {out_image_name}")
                
            # 2. Label (aneurysms.nii.gz) -> Binarize & Save
            # ADAM dataset menggunakan angka 1, 2, 3.. untuk instance aneurysm yang berbeda.
            # nnUNet membutuhkan label binary (0: background, 1: aneurysm) untuk semantic segmentation.
            try:
                mask_nii = nib.load(label_path)
                mask_data = mask_nii.get_fdata()
                
                # Binarize: semua nilai > 0 menjadi 1
                mask_data[mask_data > 0] = 1
                mask_data = mask_data.astype(np.uint8)
                
                # Save kembali
                new_mask_nii = nib.Nifti1Image(mask_data, mask_nii.affine, mask_nii.header)
                nib.save(new_mask_nii, os.path.join(OUT_LABELS_TR, out_label_name))
                print(f"[{subject}] Label SUCCESS (Binarized & Saved): {label_path} -> {out_label_name}")
            except Exception as e:
                print(f"[{subject}] Label FAILED: {str(e)}")
        else:
            print(f"[{subject}] FAILED: Missing TOF or aneurysms file")

def create_dataset_json():
    """
    Membuat file dataset.json wajib untuk nnUNet
    """
    json_dict = {
        "channel_names": {
            "0": "TOF"
        },
        "labels": {
            "background": 0,
            "aneurysm": 1
        },
        "numTraining": len(os.listdir(OUT_IMAGES_TR)),
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(OUT_DIR, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)
    print("dataset.json berhasil dibuat!")

if __name__ == "__main__":
    print("Memulai konversi dataset ADAM ke nnUNet...")
    setup_directories()
    convert_dataset()
    create_dataset_json()
    print("Konversi selesai!")
