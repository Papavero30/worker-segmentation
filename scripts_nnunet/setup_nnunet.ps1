# ==============================================================================
# Skrip Setup nnUNet untuk PC-A dan PC-B (Windows)
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Memulai Setup Environment nnUNet (RTX 5080) " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Cek Versi Python (nnUNet v2 butuh >= 3.10)
$pythonVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([decimal]$pythonVer -lt 3.10) {
    Write-Host ""
    Write-Host "[ERROR] Versi Python Anda saat ini adalah $pythonVer!" -ForegroundColor Red
    Write-Host "nnUNet v2 MEWAJIBKAN Python 3.10 ke atas." -ForegroundColor Yellow
    Write-Host "Karena Anda menggunakan pyenv, mohon install dan aktifkan Python >= 3.10 terlebih dahulu:" -ForegroundColor Yellow
    Write-Host "  pyenv install 3.10.11" -ForegroundColor Green
    Write-Host "  pyenv local 3.10.11" -ForegroundColor Green
    Write-Host ""
    Write-Host "Skrip dihentikan. Silakan ganti versi Python dan jalankan ulang skrip ini." -ForegroundColor Red
    exit 1
} else {
    Write-Host "[OK] Versi Python yang digunakan: $pythonVer" -ForegroundColor Green
}

# 2. Setup Direktori
$baseDir = "D:\Nabil\worker-segmentation"
$envDir = "$baseDir\nnUNet_env"

if (-Not (Test-Path -Path $baseDir)) {
    New-Item -ItemType Directory -Path $baseDir | Out-Null
}

Set-Location $baseDir

# 3. Buat Virtual Environment
Write-Host "`n[1/4] Membuat Virtual Environment di $envDir..." -ForegroundColor Cyan
if (-Not (Test-Path -Path $envDir)) {
    python -m venv $envDir
} else {
    Write-Host "Virtual environment sudah ada, melewati pembuatan venv." -ForegroundColor Yellow
}

# 4. Aktivasi Virtual Environment
Write-Host "[2/4] Mengaktifkan Virtual Environment..." -ForegroundColor Cyan
$activateScript = "$envDir\Scripts\Activate.ps1"
if (Test-Path -Path $activateScript) {
    & $activateScript
} else {
    Write-Host "[ERROR] Gagal menemukan script aktivasi venv!" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Melakukan upgrade pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 5. Instalasi PyTorch Nightly (CUDA 12.8) untuk RTX 5080 Blackwell
Write-Host "`n[3/4] Menginstal PyTorch (CUDA 12.8) untuk dukungan RTX 5080..." -ForegroundColor Cyan
Write-Host "Catatan: Menggunakan versi Nightly karena RTX 5080 (sm_120) belum didukung di versi stabil." -ForegroundColor Yellow
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 6. Instalasi nnUNet
Write-Host "`n[4/4] Menginstal nnUNet dari clone lokal..." -ForegroundColor Cyan
$nnunetCloneDir = "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\nnUNet"
if (Test-Path -Path $nnunetCloneDir) {
    Set-Location $nnunetCloneDir
    pip install -e .
    Set-Location $baseDir
} else {
    Write-Host "[WARNING] Folder clone nnUNet tidak ditemukan di $nnunetCloneDir" -ForegroundColor Yellow
    Write-Host "Mencoba menginstal nnUNetv2 secara langsung dari PyPI..." -ForegroundColor Yellow
    pip install nnunetv2
}

# 7. Setup Struktur Folder Environment Variabel nnUNet
Write-Host "`nMembuat struktur folder workspace nnUNet..." -ForegroundColor Cyan
$workspaceDir = "D:\Nabil\nnUNet_Workspace"
$rawDir = "$workspaceDir\nnUNet_raw"
$preprocessedDir = "$workspaceDir\nnUNet_preprocessed"
$resultsDir = "$workspaceDir\nnUNet_results"

$folders = @($rawDir, $preprocessedDir, $resultsDir)
foreach ($folder in $folders) {
    if (-Not (Test-Path -Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created: $folder" -ForegroundColor Green
    }
}

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host " SETUP SELESAI! " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "1. Jangan lupa untuk selalu mengaktifkan venv sebelum bekerja:"
Write-Host "   & `"D:\Nabil\worker-segmentation\nnUNet_env\Scripts\Activate.ps1`"" -ForegroundColor Yellow
Write-Host "2. Dataset ADAM Anda ada di: D:\Nabil\ADAM_release_subjs"
Write-Host "   Anda harus mengubah format dataset tersebut ke nnUNet Dataset Format"
Write-Host "   dan meletakkannya di: $rawDir"
Write-Host "==================================================" -ForegroundColor Cyan
