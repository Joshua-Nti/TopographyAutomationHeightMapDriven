
# 🧩 Nonplanar Slicing Pipeline Documentation
### Zurich University of Applied Sciences (ZHAW)
**Author:** [Your Name]  
**Date:** 2025-10-26

---

## 🎯 Overview

This project implements a **cross-platform automated pipeline** for **nonplanar 3D printing** using **SuperSlicer/Slic3r** and Python.  
The pipeline automates the full process — from STL analysis to final G-code transformation and positioning — with modular, configurable components.

The main objectives were:
- Reduce manual operations in nonplanar printing.
- Provide a consistent folder and processing structure.
- Achieve OS independence (macOS, Windows, Linux).
- Allow adjustable parameters via `_10config.py`.

---

## 🧠 Pipeline Architecture
 
```
Input STL
   ↓
_01analysestl.py → Detects cut heights
_02cutstl.py → Cuts STL top→bottom (configurable)
_03refinemesh.py → Refines mesh triangles
_04transformstl.py → Applies surface deformation
_05execslicer.py → Calls SuperSlicer to slice STL
_06transformgcode.py → Backtransforms & applies slowdown
_07combine.py → Merges part G-codes bottom→top
_08movegcode.py → Shifts final print on build plate
_10config.py → Central configuration + OS detection
   ↓
Final G-code
```

---

## ⚙️ Configuration: `_10config.py`

All adjustable parameters are centralized here.  
This includes paths, slicer settings, geometry tolerances, and pipeline behaviors.

### OS Detection

```python
import platform

def get_system_name():
    return platform.system()
```

- macOS → "Darwin"  
- Windows → "Windows"  
- Linux → "Linux"

### Example: Auto-select Slic3r/SuperSlicer binary

```python
def get_superslicer_binary():
    system_name = get_system_name()
    if system_name == "Darwin":
        return "/Applications/SuperSlicer.app/Contents/MacOS/SuperSlicer"
    if system_name == "Windows":
        return r"C:\Program Files\SuperSlicer\SuperSlicer.exe"
    return "superslicer"
```

### Example Config Blocks

```python
GEOMETRY_CONFIG = {
    "refine_edge_length_mm": 1.0,
    "maximal_segment_length_mm": 2.0,
    "downward_angle_deg": 10.0,
    "slow_feedrate_mm_per_min": 180.0,
    "z_desired_min_mm": 6.0,
    "xy_backtransform_shift_mm": (90.0, 90.0),
}

PIPELINE_CONFIG = {
    "apply_final_shift": True,
    "final_shift_xy_mm": (100.0, 100.0),
    "apply_backtransform_to_planar": False,
}
```

---

## 🔪 STL Cutting: `_02cutstl.py`

Cuts the STL into vertical sections (top→bottom).  
Uses `Slic3r --cut` to produce upper/lower halves iteratively.

### Key Behavior

- Reads cut heights from `cuts.txt`.
- Ignores zero-level cuts.
- Cuts in descending order (top→bottom).
- Saves segments as `<basename>_1.stl`, `<basename>_2.stl`, ... (bottom→top).

---

## 🧩 Main Controller: `_00all.py`

Central orchestrator for the full process.

### Responsibilities
1. Analyze STL → generate `cuts.txt`
2. Cut STL into parts
3. Refine and transform meshes (if nonplanar)
4. Slice parts with SuperSlicer
5. Backtransform & apply slowdown
6. Merge & shift final G-code

---

## 🚀 Shifting G-code: `_08movegcode.py`

Shifts the printed object (e.g. +100/+100 mm) only after purge and intro lines.

### Trigger Logic

1. Wait for ';LAYER_CHANGE' → arm shifting.  
2. First motion line (G0/G1) after that → start shifting.  
3. All later XY moves → shifted by (x_shift, y_shift).

### Configurable Offset

```python
x_shift, y_shift = PIPELINE_CONFIG["final_shift_xy_mm"]
```

---

## 🌍 Cross-Platform Behavior

- Binary paths determined by OS automatically.
- Path separators handled via os.path.join().
- Line endings normalized (`\n`).
- Works identically on macOS, Windows, and Linux.

---

## 💡 Summary

This pipeline provides a **complete, automated, and configurable workflow** for **nonplanar additive manufacturing**.  
It is fully **cross-platform**, modular, and designed for **research reproducibility** and **industrial scalability**.

---

📘 **ZHAW Nonplanar Slicing Pipeline**  
Developed for academic and prototyping purposes.  
For contributions or bug reports, contact the author.
