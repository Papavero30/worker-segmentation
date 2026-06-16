"""
DICOM Finder — Pindai folder untuk menemukan series TOF / MRA / angiografi
==========================================================================

MASALAH yang dipecahkan
-----------------------
File DICOM tidak bisa dikenali jenis sequence-nya (TOF/SWAN/T2/MRA) dari
nama atau bentuk file. Jenis sequence hanya ada di METADATA di dalam file.
Script ini membaca metadata semua DICOM di sebuah folder (rekursif) dan
melaporkan series mana yang merupakan TOF/MRA/angiografi -- yaitu yang
cocok untuk model segmentasi aneurysm (dilatih pada TOF).

CARA PAKAI
----------
  # Pindai satu folder (rekursif)
  python scan_dicom_finder.py --root "D:/path/ke/folder_scan_pasien"

  # Pindai beberapa folder sekaligus
  python scan_dicom_finder.py --root "D:/Pasien1" "D:/Pasien2"

  # Pindai seluruh drive D (hati-hati: bisa lama)
  python scan_dicom_finder.py --root "D:/"

OUTPUT
------
Daftar setiap folder yang berisi DICOM, dengan klasifikasi triase per
series: TOF (cocok), ANGIO? (kandidat), atau NON-ANGIO (tidak cocok).
Di akhir: ringkasan folder mana saja yang punya TOF -> siap dipakai.

PRASYARAT: pydicom (pip install pydicom). SimpleITK opsional (untuk
mengelompokkan per-series lebih akurat); jika tidak ada, script tetap
jalan dengan membaca file satu per satu.
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from pydicom import dcmread
    from pydicom.errors import InvalidDicomError
except ImportError:
    print('[ERROR] pydicom belum terpasang. Jalankan: pip install pydicom')
    sys.exit(1)


_TOF_KEYWORDS = ['tof', 't.o.f', 'time of flight', 'tone']
_ANGIO_KEYWORDS = ['mra', 'angio', 'angiography', 'cow', 'circle of willis', 'venogram', 'mrv', 'cta']


def classify(desc, scan_seq, seq_name, image_type):
    blob = ' '.join([str(x or '') for x in [desc, scan_seq, seq_name, image_type]]).lower()
    if any(k in blob for k in _TOF_KEYWORDS):
        return 'TOF'
    if any(k in blob for k in _ANGIO_KEYWORDS):
        return 'ANGIO?'
    return 'NON-ANGIO'


_MEDICAL_MODALITIES = {'MR', 'CT', 'XA', 'US', 'CR', 'DX', 'MG', 'PT', 'NM', 'OT'}
# Ekstensi file yang JELAS bukan DICOM (file aplikasi viewer, dll.)
_SKIP_EXT = {'.jar', '.dll', '.exe', '.txt', '.xml', '.json', '.properties',
             '.cfg', '.inf', '.html', '.htm', '.css', '.js', '.png', '.jpg',
             '.gif', '.ico', '.zip', '.gz', '.md', '.log', '.bat', '.sh',
             '.so', '.dylib', '.class', '.jnlp', '.lut', '.policy'}


def is_dicom(path: Path) -> bool:
    """Cek apakah file adalah DICOM citra medis yang valid.

    Strategi ketat untuk menghindari false positive dari file aplikasi
    viewer (mis. Weasis .jar): wajib ada preamble 'DICM' DAN punya
    Modality medis yang dikenal.
    """
    if path.suffix.lower() in _SKIP_EXT:
        return False
    # Wajib punya preamble DICM di offset 128 (file DICOM standar)
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            if f.read(4) != b'DICM':
                return False
    except Exception:
        return False
    # Konfirmasi punya Modality medis (bukan file DICOM-encapsulated lain)
    try:
        ds = dcmread(str(path), stop_before_pixels=True, force=True)
        modal = str(getattr(ds, 'Modality', ''))
        return modal in _MEDICAL_MODALITIES
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='Pindai folder untuk menemukan DICOM TOF/MRA')
    ap.add_argument('--root', nargs='+', required=True,
                    help='Satu atau beberapa folder yang akan dipindai (rekursif)')
    ap.add_argument('--max-files-per-series', type=int, default=1,
                    help='Cukup baca N file per series untuk metadata (default 1)')
    args = ap.parse_args()

    # series_key -> dict info; key = (folder, SeriesInstanceUID)
    series_info = {}
    series_count = defaultdict(int)
    n_scanned = 0
    n_dicom = 0

    print('Memindai... (membaca metadata, bukan piksel, jadi relatif cepat)\n')
    for root in args.root:
        root_path = Path(root)
        if not root_path.exists():
            print(f'[WARN] tidak ada: {root}')
            continue
        for p in root_path.rglob('*'):
            if not p.is_file():
                continue
            n_scanned += 1
            if not is_dicom(p):
                continue
            try:
                ds = dcmread(str(p), stop_before_pixels=True, force=True)
            except Exception:
                continue
            n_dicom += 1
            uid = str(getattr(ds, 'SeriesInstanceUID', p.parent))
            key = (str(p.parent), uid)
            series_count[key] += 1
            if key not in series_info:
                desc = str(getattr(ds, 'SeriesDescription', '?'))
                modal = str(getattr(ds, 'Modality', '?'))
                scan_seq = str(getattr(ds, 'ScanningSequence', ''))
                seq_name = str(getattr(ds, 'SequenceName', ''))
                itype = str(getattr(ds, 'ImageType', ''))
                cba = getattr(ds, 'ContrastBolusAgent', None)
                series_info[key] = {
                    'folder': str(p.parent),
                    'desc': desc, 'modal': modal,
                    'triage': classify(desc, scan_seq, seq_name, itype),
                    'contrast': 'YES' if cba else '-',
                }

    if n_dicom == 0:
        print(f'[INFO] {n_scanned} file dipindai, TIDAK ditemukan file DICOM.')
        print('       Pastikan --root menunjuk ke folder yang benar.')
        return

    # Lampirkan jumlah slice
    for key, info in series_info.items():
        info['slices'] = series_count[key]

    # Cetak per folder
    by_folder = defaultdict(list)
    for info in series_info.values():
        by_folder[info['folder']].append(info)

    tof_folders, angio_folders = [], []
    print(f'=== {n_dicom} file DICOM dalam {len(series_info)} series, {len(by_folder)} folder ===\n')
    for folder in sorted(by_folder):
        print(f'FOLDER: {folder}')
        print('  %-9s %-6s %-6s %-7s %s' % ('TRIASE', 'Modal', 'Slice', 'Contr', 'Description'))
        for info in by_folder[folder]:
            print('  %-9s %-6s %-6d %-7s %s' % (
                info['triage'], info['modal'], info['slices'], info['contrast'], info['desc'][:46]))
            if info['triage'] == 'TOF':
                tof_folders.append((folder, info['desc']))
            elif info['triage'] == 'ANGIO?':
                angio_folders.append((folder, info['desc']))
        print()

    # Ringkasan
    print('=' * 70)
    print(' RINGKASAN')
    print('=' * 70)
    if tof_folders:
        print(f' [OK] {len(tof_folders)} series TOF ditemukan (COCOK untuk model):')
        for f, d in tof_folders:
            print(f'      - {d}  @  {f}')
        print('\n      Untuk inference, jalankan test_volume_inference.py dengan')
        print('      --dicom-dir menunjuk ke folder tsb + --list-series untuk')
        print('      memilih index series TOF-nya.')
    else:
        print(' [-] Tidak ada series TOF ditemukan.')
    if angio_folders:
        print(f'\n [?] {len(angio_folders)} kandidat angiografi lain (MRA/CTA, mungkin perlu model berbeda):')
        for f, d in angio_folders:
            print(f'      - {d}  @  {f}')
    if not tof_folders and not angio_folders:
        print('\n [STOP] Tidak ada citra angiografi (TOF/MRA/CTA) sama sekali.')
        print('        Koleksi ini tidak memuat modalitas yang cocok untuk')
        print('        segmentasi aneurysm.')


if __name__ == '__main__':
    main()
