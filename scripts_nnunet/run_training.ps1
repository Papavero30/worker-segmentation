# ==============================================================================
# Skrip Run Training nnUNet 2D untuk PC-B (Windows)
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Menjalankan Pipeline nnUNet 2D " -ForegroundColor Cyan
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

$env:nnUNet_raw          = "$workspaceDir\nnUNet_raw"
$env:nnUNet_preprocessed = "$workspaceDir\nnUNet_preprocessed"
$env:nnUNet_results      = "$workspaceDir\nnUNet_results"

$env:CUDA_MODULE_LOADING      = "LAZY"
$env:nnUNet_keep_files_open   = "False"
$env:MAX_JOBS                 = "1"
$env:NINJA_MAX_JOBS           = "1"

Write-Host "  nnUNet_raw          = $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results      = $env:nnUNet_results"

# 3. Konfigurasi Dataset
$datasetId = Read-Host "`nMasukkan ID Dataset nnUNet (Contoh: 501)"
if ([string]::IsNullOrWhiteSpace($datasetId)) {
    Write-Host "[ERROR] Dataset ID tidak boleh kosong!" -ForegroundColor Red
    exit 1
}

# 4. Menu Pilihan Aksi
Write-Host "`nPilih Aksi:" -ForegroundColor Cyan
Write-Host "1. Plan & Preprocess 2D (jalankan PERTAMA KALI atau jika belum ada config 2d)"
Write-Host "2. Train 1 Fold saja"
Write-Host "3. Train semua fold (0-4) secara berurutan [REKOMENDASI]"
$choice = Read-Host "Masukkan pilihan (1/2/3)"

switch ($choice) {
    '1' {
        Write-Host "`n[Action] Plan & Preprocess konfigurasi 2D..." -ForegroundColor Green
        nnUNetv2_plan_and_preprocess -d $datasetId -c 2d --verify_dataset_integrity
    }
    '2' {
        $fold = Read-Host "Masukkan Fold (0/1/2/3/4)"
        if ([string]::IsNullOrWhiteSpace($fold)) { $fold = "0" }

        $logFile = "$env:nnUNet_results\training_fold${fold}.log"
        Write-Host "`n[Action] Training fold $fold ... (log -> $logFile)" -ForegroundColor Green
        nnUNetv2_train $datasetId 2d $fold --c *>&1 | Tee-Object -FilePath $logFile
    }
    '3' {
        Write-Host "`n[Action] Training 5-fold CV (fold 0 s/d 4) secara berurutan..." -ForegroundColor Green
        Write-Host "Training bisa resume otomatis jika PC restart (flag --c aktif)." -ForegroundColor Yellow
        Write-Host "Monitor log: Get-Content -Wait <nnUNet_results>\training_fold<N>.log`n" -ForegroundColor Yellow

        for ($f = 0; $f -lt 5; $f++) {
            $logFile = "$env:nnUNet_results\training_fold${f}.log"
            Write-Host "--------------------------------------------" -ForegroundColor DarkCyan
            Write-Host " Memulai Fold $f dari 4 -> log: $logFile" -ForegroundColor DarkCyan
            Write-Host "--------------------------------------------" -ForegroundColor DarkCyan
            nnUNetv2_train $datasetId 2d $f --c *>&1 | Tee-Object -FilePath $logFile
            Write-Host "[Fold $f] Selesai." -ForegroundColor Green
        }

        Write-Host "`n==================================================" -ForegroundColor Green
        Write-Host " SEMUA 5 FOLD SELESAI! " -ForegroundColor Green
        Write-Host " Checkpoint ada di: $env:nnUNet_results" -ForegroundColor Green
        Write-Host "==================================================" -ForegroundColor Green
    }
    default {
        Write-Host "Pilihan tidak valid. Skrip dibatalkan." -ForegroundColor Red
    }
}

Write-Host "`nSelesai." -ForegroundColor Cyan
