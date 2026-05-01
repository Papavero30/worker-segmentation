# ==============================================================================
# Skrip Run Training nnUNet 2D untuk PC-B (Windows)
# ==============================================================================

# ---------------------------------------------------------------------------
# Helper: jalankan nnUNetv2_train dan pipe output ke konsol + log file
# tanpa NativeCommandError dari PowerShell 5.1.
# Solusi: pakai cmd /c dan redirect 2>&1 di level cmd, bukan PS.
# ---------------------------------------------------------------------------
function Invoke-NnUNetTrain {
    param(
        [string]$DatasetId,
        [string]$Fold,
        [string]$LogFile
    )

    $logDir = Split-Path $LogFile -Parent
    if (-Not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

    # Header log
    $header = @"
==================================================
 nnU-Net 2D Training Log
 Dataset : $DatasetId
 Fold    : $Fold
 Started : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
==================================================
"@
    $header | Out-File -FilePath $LogFile -Encoding utf8

    Write-Host ""
    Write-Host "  [LOG] $LogFile" -ForegroundColor DarkGray
    Write-Host "  [TIP] Monitor di terminal lain:" -ForegroundColor DarkGray
    Write-Host "        Get-Content -Wait '$LogFile'" -ForegroundColor Yellow
    Write-Host ""

    # Jalankan via cmd /c supaya stderr tidak di-wrap jadi ErrorRecord PS
    # Output disalurkan ke konsol real-time DAN di-append ke log file
    $cmdLine = "nnUNetv2_train $DatasetId 2d $Fold --c"
    cmd /c "$cmdLine >> `"$LogFile`" 2>&1"

    # Tampilkan baris terakhir log sebagai summary
    Write-Host ""
    if (Test-Path $LogFile) {
        $tail = Get-Content $LogFile -Tail 8
        Write-Host "  --- Tail log (8 baris terakhir) ---" -ForegroundColor DarkCyan
        $tail | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        Write-Host "  ------------------------------------" -ForegroundColor DarkCyan
    }
}

# ---------------------------------------------------------------------------
# Helper: tampilkan status checkpoint fold yang sudah ada
# ---------------------------------------------------------------------------
function Show-CheckpointStatus {
    param([string]$ResultsDir, [string]$DatasetId)

    $modelDir = "$ResultsDir\Dataset${DatasetId}_ADAM\nnUNetTrainer__nnUNetPlans__2d"
    Write-Host ""
    Write-Host "  Status Checkpoint:" -ForegroundColor Cyan
    for ($i = 0; $i -lt 5; $i++) {
        $ckpt    = "$modelDir\fold_$i\checkpoint_final.pth"
        $latest  = "$modelDir\fold_$i\checkpoint_latest.pth"
        if (Test-Path $ckpt) {
            $ts = (Get-Item $ckpt).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            Write-Host "    fold_$i : FINAL selesai  ($ts)" -ForegroundColor Green
        } elseif (Test-Path $latest) {
            $ts = (Get-Item $latest).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            Write-Host "    fold_$i : in-progress   ($ts, checkpoint_latest.pth)" -ForegroundColor Yellow
        } else {
            Write-Host "    fold_$i : belum dimulai" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

# ==============================================================================
# MAIN
# ==============================================================================

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Menjalankan Pipeline nnUNet 2D - PC-B (RTX 5080)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Cek & Aktivasi venv
$baseDir = "D:\Nabil\worker-segmentation"
$envDir  = "$baseDir\nnUNet_env"
$activateScript = "$envDir\Scripts\Activate.ps1"

if (-Not (Test-Path $activateScript)) {
    Write-Host "[ERROR] Virtual Environment belum dibuat! Jalankan setup_nnunet.ps1 dulu." -ForegroundColor Red
    exit 1
}
Write-Host "[1/3] Mengaktifkan Virtual Environment..." -ForegroundColor Cyan
& $activateScript

# 2. Setup env vars
Write-Host "[2/3] Mengatur Environment Variables nnUNet..." -ForegroundColor Cyan
$workspaceDir = "D:\Nabil\nnUNet_Workspace"

$env:nnUNet_raw          = "$workspaceDir\nnUNet_raw"
$env:nnUNet_preprocessed = "$workspaceDir\nnUNet_preprocessed"
$env:nnUNet_results      = "$workspaceDir\nnUNet_results"
$env:CUDA_MODULE_LOADING = "LAZY"
$env:nnUNet_keep_files_open = "False"
$env:MAX_JOBS            = "1"
$env:NINJA_MAX_JOBS      = "1"

Write-Host "  nnUNet_raw          = $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results      = $env:nnUNet_results"

# 3. Dataset ID
$datasetId = Read-Host "`n[3/3] Masukkan ID Dataset nnUNet (Contoh: 501)"
if ([string]::IsNullOrWhiteSpace($datasetId)) {
    Write-Host "[ERROR] Dataset ID tidak boleh kosong!" -ForegroundColor Red
    exit 1
}

# Tampilkan status checkpoint sebelum aksi
Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId

# 4. Menu
Write-Host "Pilih Aksi:" -ForegroundColor Cyan
Write-Host "  1. Plan & Preprocess 2D (PERTAMA KALI atau jika config 2d belum ada)"
Write-Host "  2. Train 1 Fold saja"
Write-Host "  3. Train semua fold 0-4 secara berurutan [REKOMENDASI]"
Write-Host "  4. Cek status checkpoint saja"
$choice = Read-Host "Masukkan pilihan (1/2/3/4)"

switch ($choice) {

    '1' {
        Write-Host "`n[Action] Plan & Preprocess konfigurasi 2D..." -ForegroundColor Green
        $ppLog = "$env:nnUNet_results\plan_and_preprocess_2d.log"
        Write-Host "  [LOG] $ppLog" -ForegroundColor DarkGray
        cmd /c "nnUNetv2_plan_and_preprocess -d $datasetId -c 2d --verify_dataset_integrity >> `"$ppLog`" 2>&1"
        Write-Host "[SELESAI] Plan & Preprocess 2D done." -ForegroundColor Green
    }

    '2' {
        $fold = Read-Host "Masukkan Fold (0/1/2/3/4)"
        if ([string]::IsNullOrWhiteSpace($fold)) { $fold = "0" }

        $logFile = "$env:nnUNet_results\training_fold${fold}.log"
        Write-Host ""
        Write-Host "================================================" -ForegroundColor DarkCyan
        Write-Host "  Training Fold $fold" -ForegroundColor DarkCyan
        Write-Host "================================================" -ForegroundColor DarkCyan
        $foldStart = Get-Date

        Invoke-NnUNetTrain -DatasetId $datasetId -Fold $fold -LogFile $logFile

        $elapsed = (Get-Date) - $foldStart
        Write-Host ""
        Write-Host "[Fold $fold] Selesai. Durasi: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId
    }

    '3' {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor DarkCyan
        Write-Host "  Training 5-Fold CV (fold 0 s/d 4)" -ForegroundColor DarkCyan
        Write-Host "  Flag --c aktif: resume otomatis jika PC restart" -ForegroundColor DarkCyan
        Write-Host "================================================" -ForegroundColor DarkCyan

        $totalStart = Get-Date

        for ($f = 0; $f -lt 5; $f++) {
            $logFile   = "$env:nnUNet_results\training_fold${f}.log"
            $foldStart = Get-Date

            # Cek apakah fold ini sudah final (skip jika sudah)
            $modelDir = "$env:nnUNet_results\Dataset${datasetId}_ADAM\nnUNetTrainer__nnUNetPlans__2d"
            $ckptFinal = "$modelDir\fold_$f\checkpoint_final.pth"
            if (Test-Path $ckptFinal) {
                Write-Host ""
                Write-Host "  [SKIP] fold_$f sudah FINAL ($ckptFinal). Lanjut ke fold berikutnya." -ForegroundColor Green
                continue
            }

            Write-Host ""
            Write-Host "  ============================================" -ForegroundColor Cyan
            Write-Host "  Fold $f / 4  |  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
            Write-Host "  ============================================" -ForegroundColor Cyan

            Invoke-NnUNetTrain -DatasetId $datasetId -Fold $f -LogFile $logFile

            $foldElapsed = (Get-Date) - $foldStart
            Write-Host ""
            Write-Host "  [Fold $f] Selesai dalam $($foldElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId

            # Estimasi sisa waktu
            $doneFolds  = $f + 1
            $avgSeconds = ((Get-Date) - $totalStart).TotalSeconds / $doneFolds
            $remaining  = [int]($avgSeconds * (5 - $doneFolds))
            $eta        = (Get-Date).AddSeconds($remaining).ToString('HH:mm')
            Write-Host "  [ETA] Estimasi selesai semua fold: ~$eta (sisa $([math]::Round($remaining/3600,1)) jam)" -ForegroundColor Yellow
        }

        $totalElapsed = (Get-Date) - $totalStart
        Write-Host ""
        Write-Host "==================================================" -ForegroundColor Green
        Write-Host "  SEMUA 5 FOLD SELESAI!" -ForegroundColor Green
        Write-Host "  Total durasi : $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Write-Host "  Checkpoint   : $env:nnUNet_results" -ForegroundColor Green
        Write-Host "==================================================" -ForegroundColor Green
        Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId
    }

    '4' {
        Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId
    }

    default {
        Write-Host "Pilihan tidak valid." -ForegroundColor Red
    }
}

Write-Host "`nSelesai." -ForegroundColor Cyan
