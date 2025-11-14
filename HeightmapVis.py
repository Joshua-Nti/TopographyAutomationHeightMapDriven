# heightmapVis.py
# Real-time interactive 3D visualization of stored heightmap_*.npy files

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plot


def load_and_show_heightmap(npy_path):
    """
    Loads a stored heightmap .npy file and visualizes it in interactive 3D.
    - Scroll, pan, rotate freely
    - Shows heightmap as a 3D surface
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Heightmap file not found: {npy_path}")

    print(f"[heightmapVis] Loading heightmap: {npy_path}")
    H = np.load(npy_path)

    ny, nx = H.shape
    X = np.arange(nx)
    Y = np.arange(ny)
    X, Y = np.meshgrid(X, Y)

    print("[heightmapVis] Shape:", H.shape)

    # --- 3D Visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, H,
        cmap='viridis',
        edgecolor='none',
        linewidth=0,
        antialiased=True
    )

    ax.set_title(f"3D Heightmap Visualization\n{os.path.basename(npy_path)}")
    ax.set_xlabel("X grid index")
    ax.set_ylabel("Y grid index")
    ax.set_zlabel("Height (mm)")

    fig.colorbar(surf, ax=ax, shrink=0.6, label="Height (mm)")

    print("[heightmapVis] Rendering interactive window...")
    plt.show()


if __name__ == "__main__":
    # EXAMPLES  
    # To view a model's heightmap, update with that file OR pass via CLI.

    default_file = os.path.join("test","stl_tf", "heightmap_test_5.npy")

    if os.path.exists(default_file):
        load_and_show_heightmap(default_file)
    else:
        print("Default file not found. Please specify a valid npy path:")
        print("  python heightmapVis.py stl_tf/heightmap_test_4.npy")

    # Command-line support
    import sys
    if len(sys.argv) == 2:
        load_and_show_heightmap(sys.argv[1])
