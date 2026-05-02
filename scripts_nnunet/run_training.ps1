# ==============================================================================
# Skrip Run Training nnUNet 2D untuk PC-B (Windows)
#
# FIXES:
# - Isolasi venv dari Anaconda base (deactivate conda dulu)
# - nnUNet_n_proc_DA=0 untuk hindari WinError 6 (multiprocessing handle invalid)
# - Cek exit code nnUNetv2_train, STOP loop kalau crash
# - Skip fold yang sudah punya checkpoint_final.pth
# ==============================================================================

# ---------------------------------------------------------------------------
# Helper: jalankan nnUNetv2_train + cek exit code
# Return $true kalau benar-benar selesai (checkpoint_final.pth ada),
# $false kalau crash/incomplete.
# ---------------------------------------------------------------------------
function Invoke-NnUNetTrain {
    param(
        [string]$DatasetId,
        [string]$Fold,
        [string]$LogFile,
        [string]$ModelDir
    )

    $logDir = Split-Path $LogFile -Parent
    if (-Not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

    $header = @"
==================================================
 nnU-Net 2D Training Log
 Dataset : $DatasetId
 Fold    : $Fold
 Started : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
 nnUNet_n_proc_DA = $env:nnUNet_n_proc_DA
==================================================
"@
    $header | Out-File -FilePath $LogFile -Encoding utf8

    Write-Host ""
    Write-Host "  [LOG] $LogFile" -ForegroundColor DarkGray
    Write-Host "  [TIP] Monitor di terminal lain:" -ForegroundColor DarkGray
    Write-Host "        Get-Content -Wait '$LogFile'" -ForegroundColor Yellow
    Write-Host ""

    # Jalankan via cmd /c, redirect stderr supaya PS tidak salah parse
    $cmdLine = "nnUNetv2_train $DatasetId 2d $Fold --c"
    cmd /c "$cmdLine >> `"$LogFile`" 2>&1"
    $cmdExit = $LASTEXITCODE

    # Tail log
    Write-Host ""
    if (Test-Path $LogFile) {
        $tail = Get-Content $LogFile -Tail 10
        Write-Host "  --- Tail log (10 baris terakhir) ---" -ForegroundColor DarkCyan
        $tail | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        Write-Host "  ------------------------------------" -ForegroundColor DarkCyan
    }

    # Verify final checkpoint ada
    $ckptFinal = Join-Path $ModelDir "fold_$Fold\checkpoint_final.pth"
    $hasFinal = Test-Path $ckptFinal

    Write-Host ""
    Write-Host "  [VERIFY] cmd exit code   : $cmdExit" -ForegroundColor $(if ($cmdExit -eq 0) {'Green'} else {'Red'})
    Write-Host "  [VERIFY] checkpoint_final: $(if ($hasFinal) {'EXISTS'} else {'MISSING'})" -ForegroundColor $(if ($hasFinal) {'Green'} else {'Red'})

    return ($cmdExit -eq 0 -and $hasFinal)
}

function Show-CheckpointStatus {
    param([string]$ResultsDir, [string]$DatasetId)

    $modelDir = "$ResultsDir\Dataset${DatasetId}_ADAM\nnUNetTrainer__nnUNetPlans__2d"
    Write-Host ""
    Write-Host "  Status Checkpoint:" -ForegroundColor Cyan
    for ($i = 0; $i -lt 5; $i++) {
        $ckpt   = "$modelDir\fold_$i\checkpoint_final.pth"
        $latest = "$modelDir\fold_$i\checkpoint_latest.pth"
        if (Test-Path $ckpt) {
            $ts = (Get-Item $ckpt).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            Write-Host "    fold_$i : FINAL ($ts)" -ForegroundColor Green
        } elseif (Test-Path $latest) {
            $ts = (Get-Item $latest).LastWriteTime.ToString('yyyy-MM-dd HH:mm')
            Write-Host "    fold_$i : in-progress ($ts)" -ForegroundColor Yellow
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
Write-Host "  Pipeline nnUNet 2D - PC-B (RTX 5080)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. ISOLASI dari Anaconda base
# Kalau (base) Anaconda aktif, multiprocessing.synchronize akan resolve ke
# E:\Anaconda\Lib (Python 3.13) padahal venv pakai 3.10 -> WinError 6 crash.
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "[WARN] Anaconda environment aktif: $env:CONDA_DEFAULT_ENV" -ForegroundColor Yellow
    Write-Host "       Menonaktifkan untuk mencegah konflik multiprocessing..." -ForegroundColor Yellow
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda deactivate 2>&1 | Out-Null
        conda deactivate 2>&1 | Out-Null   # double untuk keluar (base)
    }
    # Buang juga PYTHONPATH/PYTHONHOME warisan conda
    Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
    Write-Host "[OK] Conda deactivated." -ForegroundColor Green
}

# 2. Aktivasi venv
$baseDir = "D:\Nabil\worker-segmentation"
$envDir  = "$baseDir\nnUNet_env"
$activateScript = "$envDir\Scripts\Activate.ps1"

if (-Not (Test-Path $activateScript)) {
    Write-Host "[ERROR] Virtual Environment belum dibuat! Jalankan setup_nnunet.ps1 dulu." -ForegroundColor Red
    exit 1
}
Write-Host "[1/3] Mengaktifkan Virtual Environment..." -ForegroundColor Cyan
& $activateScript

# Verify Python source
$pyExe = (Get-Command python).Source
Write-Host "  python: $pyExe" -ForegroundColor DarkGray
if ($pyExe -notlike "*nnUNet_env*") {
    Write-Host "[ERROR] Python BUKAN dari venv! ($pyExe)" -ForegroundColor Red
    Write-Host "       Buka shell baru dan jalankan ulang script ini." -ForegroundColor Yellow
    exit 1
}

# 3. Setup env vars
Write-Host "[2/3] Mengatur Environment Variables nnUNet..." -ForegroundColor Cyan
$workspaceDir = "D:\Nabil\nnUNet_Workspace"

$env:nnUNet_raw          = "$workspaceDir\nnUNet_raw"
$env:nnUNet_preprocessed = "$workspaceDir\nnUNet_preprocessed"
$env:nnUNet_results      = "$workspaceDir\nnUNet_results"

# CRITICAL FIX untuk WinError 6: disable multi-thread data augmentation worker.
# Default 12 worker bikin race condition di Windows multiprocessing.
# Dengan nnUNet_n_proc_DA=0, augmentation jalan single-threaded (lebih stabil,
# sedikit lebih lambat tapi GPU sudah jadi bottleneck).
$env:nnUNet_n_proc_DA       = "0"
$env:nnUNet_keep_files_open = "False"
$env:CUDA_MODULE_LOADING    = "LAZY"
$env:MAX_JOBS               = "1"
$env:NINJA_MAX_JOBS         = "1"

Write-Host "  nnUNet_raw          = $env:nnUNet_raw"
Write-Host "  nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "  nnUNet_results      = $env:nnUNet_results"
Write-Host "  nnUNet_n_proc_DA    = $env:nnUNet_n_proc_DA  (0 = single-threaded augmenter, fix WinError 6)" -ForegroundColor DarkGray

# 4. Dataset ID
$datasetId = Read-Host "`n[3/3] Masukkan ID Dataset nnUNet (Contoh: 501)"
if ([string]::IsNullOrWhiteSpace($datasetId)) {
    Write-Host "[ERROR] Dataset ID tidak boleh kosong!" -ForegroundColor Red
    exit 1
}

Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId

# 5. Menu
Write-Host "Pilih Aksi:" -ForegroundColor Cyan
Write-Host "  1. Plan & Preprocess 2D"
Write-Host "  2. Train 1 Fold (untuk test cepat)"
Write-Host "  3. Train semua fold 0-4 [STOP-ON-FAIL]"
Write-Host "  4. Cek status checkpoint saja"
$choice = Read-Host "Masukkan pilihan (1/2/3/4)"

switch ($choice) {

    '1' {
        Write-Host "`n[Action] Plan & Preprocess konfigurasi 2D..." -ForegroundColor Green
        $ppLog = "$env:nnUNet_results\plan_and_preprocess_2d.log"
        Write-Host "  [LOG] $ppLog" -ForegroundColor DarkGray
        cmd /c "nnUNetv2_plan_and_preprocess -d $datasetId -c 2d --verify_dataset_integrity >> `"$ppLog`" 2>&1"
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Host "[OK] Plan & Preprocess selesai." -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Plan & Preprocess GAGAL (exit $exitCode). Cek log: $ppLog" -ForegroundColor Red
        }
    }

    '2' {
        $fold = Read-Host "Masukkan Fold (0/1/2/3/4)"
        if ([string]::IsNullOrWhiteSpace($fold)) { $fold = "0" }

        $logFile  = "$env:nnUNet_results\training_fold${fold}.log"
        $modelDir = "$env:nnUNet_results\Dataset${datasetId}_ADAM\nnUNetTrainer__nnUNetPlans__2d"

        Write-Host ""
        Write-Host "================================================" -ForegroundColor DarkCyan
        Write-Host "  Training Fold $fold" -ForegroundColor DarkCyan
        Write-Host "================================================" -ForegroundColor DarkCyan
        $foldStart = Get-Date
        $ok = Invoke-NnUNetTrain -DatasetId $datasetId -Fold $fold -LogFile $logFile -ModelDir $modelDir
        $elapsed = (Get-Date) - $foldStart

        if ($ok) {
            Write-Host "[Fold $fold] SUCCESS dalam $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        } else {
            Write-Host "[Fold $fold] FAILED setelah $($elapsed.ToString('hh\:mm\:ss')). Cek log untuk detail." -ForegroundColor Red
        }
        Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId
    }

    '3' {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor DarkCyan
        Write-Host "  Training 5-Fold CV (STOP-ON-FAIL)" -ForegroundColor DarkCyan
        Write-Host "  Flag --c aktif: resume otomatis setelah crash" -ForegroundColor DarkCyan
        Write-Host "================================================" -ForegroundColor DarkCyan

        $modelDir   = "$env:nnUNet_results\Dataset${datasetId}_ADAM\nnUNetTrainer__nnUNetPlans__2d"
        $totalStart = Get-Date

        for ($f = 0; $f -lt 5; $f++) {
            $logFile = "$env:nnUNet_results\training_fold${f}.log"

            # Skip fold yang sudah final
            $ckptFinal = "$modelDir\fold_$f\checkpoint_final.pth"
            if (Test-Path $ckptFinal) {
                Write-Host ""
                Write-Host "  [SKIP] fold_$f sudah FINAL." -ForegroundColor Green
                continue
            }

            Write-Host ""
            Write-Host "  ============================================" -ForegroundColor Cyan
            Write-Host "  Fold $f / 4  |  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
            Write-Host "  ============================================" -ForegroundColor Cyan

            $foldStart = Get-Date
            $ok = Invoke-NnUNetTrain -DatasetId $datasetId -Fold $f -LogFile $logFile -ModelDir $modelDir
            $foldElapsed = (Get-Date) - $foldStart

            if (-not $ok) {
                Write-Host ""
                Write-Host "  [STOP] Fold $f FAILED setelah $($foldElapsed.ToString('hh\:mm\:ss'))." -ForegroundColor Red
                Write-Host "         Loop dihentikan. Cek log: $logFile" -ForegroundColor Red
                Write-Host "         Setelah fix masalah, jalankan lagi script ini -> pilih 3" -ForegroundColor Yellow
                Write-Host "         (fold yang sudah final akan di-skip otomatis)" -ForegroundColor Yellow
                Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId
                exit 1
            }

            Write-Host "  [Fold $f] SUCCESS dalam $($foldElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            Show-CheckpointStatus -ResultsDir $env:nnUNet_results -DatasetId $datasetId

            $doneFolds  = $f + 1
            $avgSeconds = ((Get-Date) - $totalStart).TotalSeconds / $doneFolds
            $remaining  = [int]($avgSeconds * (5 - $doneFolds))
            $eta        = (Get-Date).AddSeconds($remaining).ToString('HH:mm dd-MMM')
            Write-Host "  [ETA] Estimasi semua fold selesai: $eta" -ForegroundColor Yellow
        }

        $totalElapsed = (Get-Date) - $totalStart
        Write-Host ""
        Write-Host "==================================================" -ForegroundColor Green
        Write-Host "  SEMUA 5 FOLD SUCCESS!" -ForegroundColor Green
        Write-Host "  Total durasi : $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        Write-Host "  Checkpoint   : $modelDir" -ForegroundColor Green
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
