# ==============================================================================
# Skrip Run Training nnUNet untuk PC-A dan PC-B (Windows)
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Menjalankan Pipeline nnUNet " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Konfigurasi Path dan Aktivasi Venv
$baseDir = "D:\Nabil\worker-segmentation"
$envDir = "$baseDir\nnUNet_env"
$activateScript = "$envDir\Scripts\Activate.ps1"

if (-Not (Test-Path -Path $activateScript)) {
    Write-Host "[ERROR] Virtual Environment belum dibuat!" -ForegroundColor Red
    Write-Host "Jalankan skrip setup_nnunet.ps1 terlebih dahulu." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/3] Mengaktifkan Virtual Environment..." -ForegroundColor Cyan
& $activateScript

# 2. Setup Environment Variables untuk nnUNet
Write-Host "[2/3] Mengatur Environment Variables nnUNet..." -ForegroundColor Cyan
$workspaceDir = "D:\Nabil\nnUNet_Workspace"

$env:nnUNet_raw = "$workspaceDir\nnUNet_raw"
$env:nnUNet_preprocessed = "$workspaceDir\nnUNet_preprocessed"
$env:nnUNet_results = "$workspaceDir\nnUNet_results"

# Optimasi CUDA untuk mencegah Out of Memory & PTXAS Memory Allocation Failure di RTX 5080
$env:CUDA_MODULE_LOADING = "LAZY"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:nnUNet_keep_files_open = "False"

Write-Host "  nnUNet_raw          = $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results      = $env:nnUNet_results"
Write-Host "  [OPT] CUDA_MODULE_LOADING       = $env:CUDA_MODULE_LOADING"
Write-Host "  [OPT] PYTORCH_CUDA_ALLOC_CONF   = $env:PYTORCH_CUDA_ALLOC_CONF"

# 3. Konfigurasi ID Dataset
$datasetId = Read-Host "`nMasukkan ID Dataset nnUNet (Contoh: 100 atau 500)"

if ([string]::IsNullOrWhiteSpace($datasetId)) {
    Write-Host "[ERROR] Dataset ID tidak boleh kosong!" -ForegroundColor Red
    exit 1
}

# 4. Menu Pilihan Aksi
Write-Host "`nPilih Aksi yang ingin dijalankan:" -ForegroundColor Cyan
Write-Host "1. Plan & Preprocess (Jalankan ini PERTAMA KALI setelah convert dataset)"
Write-Host "2. Train Model (Jalankan ini setelah preprocessing selesai)"
Write-Host "3. Jalankan KEDUANYA secara berurutan"
$choice = Read-Host "Masukkan pilihan (1/2/3)"

switch ($choice) {
    '1' {
        Write-Host "`n[Action] Menjalankan nnUNetv2_plan_and_preprocess..." -ForegroundColor Green
        nnUNetv2_plan_and_preprocess -d $datasetId --verify_dataset_integrity
    }
    '2' {
        $fold = Read-Host "Masukkan Fold untuk training (0/1/2/3/4 atau 'all' untuk 5-fold CV)"
        if ([string]::IsNullOrWhiteSpace($fold)) { $fold = "0" }
        Write-Host "`n[Action] Menjalankan nnUNetv2_train pada dataset $datasetId fold $fold..." -ForegroundColor Green
        nnUNetv2_train $datasetId 3d_fullres $fold
    }
    '3' {
        Write-Host "`n[Action 1/2] Menjalankan nnUNetv2_plan_and_preprocess..." -ForegroundColor Green
        nnUNetv2_plan_and_preprocess -d $datasetId --verify_dataset_integrity
        
        Write-Host "`n[Action 2/2] Menjalankan nnUNetv2_train (Fold 0)..." -ForegroundColor Green
        nnUNetv2_train $datasetId 3d_fullres 0
    }
    default {
        Write-Host "Pilihan tidak valid. Skrip dibatalkan." -ForegroundColor Red
    }
}

Write-Host "`nSelesai." -ForegroundColor Cyan
