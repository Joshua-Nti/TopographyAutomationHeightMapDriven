# heightmapvisual.py — visualize all saved heightmaps (2D + 3D) with smooth model outline
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

HEIGHTMAP_DIR = "heightmaps"

# Try to use scikit-image for smooth contours; fall back gracefully if missing
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[heightmapvisual] scikit-image not found; using rougher matplotlib contour as fallback.")


def _get_footprint_mask(data, DZ):
    """
    Try to recover a model-footprint mask from the .npz file.
    If none is stored, fall back to a heuristic based on DZ.
    Returns a boolean array with True = inside model footprint.
    """
    # 1) Prefer explicit masks saved by the pipeline
    for key in ["mask", "footprint", "inside_mask", "valid"]:
        if key in data:
            mask = data[key].astype(bool)
            if mask.shape == DZ.shape:
                return mask

    # 2) Heuristic fallback: use NaN or near-min regions
    if np.isnan(DZ).any():
        mask = np.isfinite(DZ)
    else:
        dz_min = float(np.min(DZ))
        eps = 1e-3
        mask = DZ > (dz_min + eps)

    return mask


def plot_heightmap_2d(X, Y, DZ, mask, out_path):
    plt.figure(figsize=(7, 6))
    plt.title("Heightmap Δz(x,y) — 2D Heatmap")

    im = plt.imshow(
        DZ,
        origin="lower",
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im, label="Δz [mm]")

    # --- Smooth model outline on top of heatmap ---
    try:
        if HAS_SKIMAGE:
            ny, nx = mask.shape
            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()

            # Marching-squares contour at 0.5 of the binary mask
            contours = measure.find_contours(mask.astype(float), 0.5)

            for c in contours:
                # c[:, 0] = row index (y), c[:, 1] = col index (x)
                xs = x_min + (c[:, 1] / (nx - 1)) * (x_max - x_min)
                ys = y_min + (c[:, 0] / (ny - 1)) * (y_max - y_min)
                plt.plot(xs, ys, color="white", linewidth=1.4, solid_joinstyle="round")
        else:
            # Fallback: still show something if skimage isn't available
            plt.contour(
                X,
                Y,
                mask.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=1.4,
            )
    except Exception as e:
        print("  (Warning) Could not draw smooth footprint outline in 2D:", e)

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("  Saved 2D heatmap →", out_path)


def plot_heightmap_3d(X, Y, DZ, out_path, elev=35, azim=135):
    """
    3D surface of the heightmap only (no outline projected).
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        DZ,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=2,
        cstride=2,
    )

    ax.set_title("Heightmap Δz(x,y) — 3D Surface")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Δz [mm]")

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    plt.savefig(out_path, dpi=220)
    print("  Saved 3D surface →", out_path)

    # Uncomment if you want an interactive window:
    # plt.show()

    plt.close()


def process_all_heightmaps():
    if not os.path.isdir(HEIGHTMAP_DIR):
        print("No 'heightmaps/' directory found.")
        return

    files = [f for f in os.listdir(HEIGHTMAP_DIR) if f.endswith("_heightmap.npz")]
    if not files:
        print("No *_heightmap.npz files found in 'heightmaps/'.")
        return

    for fname in files:
        path = os.path.join(HEIGHTMAP_DIR, fname)
        print("\nProcessing:", path)

        data = np.load(path)
        X = data["X"]
        Y = data["Y"]
        DZ = data["DZ"]

        # Build or load footprint mask
        mask = _get_footprint_mask(data, DZ)

        base = fname.replace(".npz", "")
        out_2d = os.path.join(HEIGHTMAP_DIR, base + "_2D.png")
        out_3d = os.path.join(HEIGHTMAP_DIR, base + "_3D.png")

        plot_heightmap_2d(X, Y, DZ, mask, out_2d)
        plot_heightmap_3d(X, Y, DZ, out_3d)


if __name__ == "__main__":
    process_all_heightmaps()
