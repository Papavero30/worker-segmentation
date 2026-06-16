"""
Visualisasi hasil segmentasi yang lebih informatif:
- Overlay ground truth (HIJAU) dan prediksi (MERAH) bersamaan.
- Crop/zoom ke area sekitar aneurysm supaya terlihat jelas.
- Dilatasi tipis pada mask supaya piksel kecil tetap terlihat.
- Side-by-side comparison: full slice + zoom.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation, find_objects
except ImportError as e:
    print(f'[ERROR] dep belum terpasang: {e}')
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tof', required=True, help='Path TOF NIfTI')
    ap.add_argument('--pred', required=True, help='Path predicted mask NIfTI')
    ap.add_argument('--gt', required=True, help='Path ground truth NIfTI (label 1 = aneurysm)')
    ap.add_argument('--out-dir', default='./hasil_visual', help='Folder output')
    ap.add_argument('--zoom-pad', type=int, default=40, help='Padding piksel utk zoom')
    ap.add_argument('--dilate', type=int, default=2, help='Dilatasi mask supaya terlihat')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol = nib.load(args.tof).get_fdata().astype(np.float32)
    pred = nib.load(args.pred).get_fdata().astype(np.uint8)
    gt_raw = nib.load(args.gt).get_fdata().astype(np.uint8)
    gt = (gt_raw == 1).astype(np.uint8)  # hanya target aneurysm (label 1)

    # Cari axis slice (axis terkecil)
    axis_z = int(np.argmin(vol.shape))
    vol_zyx = np.moveaxis(vol, axis_z, 0)
    pred_zyx = np.moveaxis(pred, axis_z, 0)
    gt_zyx = np.moveaxis(gt, axis_z, 0)

    # Slice mana saja yang ada GT atau prediksi
    slices_gt = set(np.where((gt_zyx > 0).reshape(gt_zyx.shape[0], -1).any(axis=1))[0])
    slices_pred = set(np.where((pred_zyx > 0).reshape(pred_zyx.shape[0], -1).any(axis=1))[0])
    relevant = sorted(slices_gt | slices_pred)

    print(f'Volume shape (ZYX): {vol_zyx.shape}')
    print(f'Slice ber-GT       : {sorted(slices_gt)}')
    print(f'Slice ber-prediksi : {sorted(slices_pred)}')
    print(f'Total slice rendered: {len(relevant)}')

    # Bbox global utk zoom (gabungan GT + pred)
    combined = (gt_zyx | pred_zyx) > 0
    nz = np.argwhere(combined)
    if len(nz) == 0:
        print('[STOP] Tidak ada GT maupun prediksi di volume manapun.')
        sys.exit(0)
    ymin, xmin = nz[:, 1].min(), nz[:, 2].min()
    ymax, xmax = nz[:, 1].max(), nz[:, 2].max()
    pad = args.zoom_pad
    ymin = max(0, ymin - pad); xmin = max(0, xmin - pad)
    ymax = min(vol_zyx.shape[1], ymax + pad); xmax = min(vol_zyx.shape[2], xmax + pad)
    print(f'Bbox zoom (y,x)    : ({ymin},{xmin}) -> ({ymax},{xmax})')

    # Untuk visibilitas, dilatasi tipis pada mask
    iters = max(1, args.dilate)

    for z in relevant:
        img = vol_zyx[z]
        gt_s = gt_zyx[z]
        pred_s = pred_zyx[z]
        gt_d = binary_dilation(gt_s, iterations=iters)
        pred_d = binary_dilation(pred_s, iterations=iters)
        fg_gt = int(gt_s.sum())
        fg_pred = int(pred_s.sum())

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Full slice
        axes[0].imshow(img, cmap='gray')
        overlay_full = np.zeros((*img.shape, 4))
        overlay_full[gt_d > 0] = [0, 1, 0, 0.55]    # GT hijau
        overlay_full[pred_d > 0] = [1, 0, 0, 0.55]  # Pred merah
        overlay_full[(gt_d > 0) & (pred_d > 0)] = [1, 1, 0, 0.65]  # overlap kuning
        axes[0].imshow(overlay_full)
        axes[0].set_title(f'Slice {z} — Full View\nGT={fg_gt}px (hijau), Pred={fg_pred}px (merah), Overlap (kuning)')
        axes[0].axis('off')

        # Zoom
        img_z = img[ymin:ymax, xmin:xmax]
        gt_zoom = gt_d[ymin:ymax, xmin:xmax]
        pred_zoom = pred_d[ymin:ymax, xmin:xmax]
        axes[1].imshow(img_z, cmap='gray')
        overlay_zoom = np.zeros((*img_z.shape, 4))
        overlay_zoom[gt_zoom > 0] = [0, 1, 0, 0.55]
        overlay_zoom[pred_zoom > 0] = [1, 0, 0, 0.55]
        overlay_zoom[(gt_zoom > 0) & (pred_zoom > 0)] = [1, 1, 0, 0.65]
        axes[1].imshow(overlay_zoom)
        axes[1].set_title(f'Slice {z} — Zoom (Circle of Willis area)\nHijau=GT, Merah=Pred, Kuning=Overlap')
        axes[1].axis('off')

        out_path = out_dir / f'compare_slice_{z:03d}.png'
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'[SAVED] {out_path}')

    print(f'\nSelesai. {len(relevant)} gambar perbandingan disimpan di {out_dir}')


if __name__ == '__main__':
    main()
