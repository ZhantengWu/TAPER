# TAPER: Time‑Aware Pipeline for Efficient, Robust QR/Barcode Decoding

> Training‑free, **budgeted** decoding that adapts compute to image difficulty and **stops at first success**.

TAPER turns a standard QR/Barcode reader into an **anytime system**: it uses lightweight cues (contrast / tilt / target size) to **conditionally** trigger enhancement steps (thresholding, rotation, deskew, upsampling), attempts a decode after each step, and **respects a per‑image time budget**. Accuracy increases with budget while easy cases exit early.

- Paper / method overview: see the included PDF. fileciteturn0file0  
- Core retry algorithm (time‑aware conditional pipeline): `decode/retry.py`. fileciteturn0file1  
- Backends (ZXing, Pyzbar, QReader, Dynamsoft): `decode/backend.py`. fileciteturn0file3  
- CLI runner for batches: `run_folder_ours.py`. fileciteturn0file2

---

##  Key Features

- **Budgeted decoding**: wall‑clock time budget per image (e.g., 200–500 ms).  
- **Early‑stop anytime behavior**: return as soon as a backend succeeds.  
- **Lightweight cues**: fast, on‑device estimates of contrast, tilt/skew, and target size.  
- **Conditional steps**: trigger only the fixes that are likely to help (threshold / rotate / deskew / upsample).  
- **Pluggable backends**: ZXing‑CPP (default), Pyzbar/ZBar, QReader, Dynamsoft DBR.  
- **Reproducible evaluation**: logs per‑stage timings, success/timeout, and stage‑of‑success for tables and curves.

See the PDF for the full method, ablations, and budgeted evaluation protocol. fileciteturn0file0

---

## Repository Layout (reference)

```
taper/
├─ README.md
├─ run_folder_ours.py            # CLI entry (batch evaluation)  fileciteturn0file2
├─ decode/
│  ├─ backend.py                 # Backends (ZXing, Pyzbar, QReader, DBR)  fileciteturn0file3
│  └─ retry.py                   # Budgeted conditional retry core            fileciteturn0file1
├─ preprocess/
│  └─ ops.py                     # limit_image_size, auto_threshold, rotate_try, deskew_light, etc.
├─ utils/
│  ├─ io.py                      # list_images, save_json, load_yaml, etc.
│  └─ timer.py                   # now_ms, elapsed_ms, Timer
└─ configs/
   └─ default.yaml               # thresholds/angles/scales; budget; enabled steps
```

> Note: `preprocess/ops.py`, `utils/io.py`, `utils/timer.py` are referenced by the code. Ensure these exist in your repo (names can be adapted as long as imports match). fileciteturn0file1

---

## Installation

Tested on Python 3.9–3.11.

```bash
# QR Code Robust Reading - Python Dependencies
# Core image processing and computer vision
opencv-python>=4.5.1,<4.13.0
numpy>=1.21.0,<2.0.0

# QR/Barcode decoding libraries
pyzbar>=0.1.9
qreader>=3.14
dbr>=10.0.0

# Data processing and analysis
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Configuration and I/O
PyYAML>=6.0
pathlib2>=2.3.0

# Progress bars and utilities
tqdm>=4.62.0

# Optional: ZXing for additional decoding support
# Note: Requires Java Runtime Environment (JRE)
# zxing>=0.9.3

# Optional: Advanced visualization
# plotly>=5.0.0

# Development and testing (optional)
pytest>=6.0.0
pytest-cov>=2.12.0

# Note: Some packages may require additional system dependencies:
# - pyzbar: requires zbar library (libzbar0 on Ubuntu, zbar-tools on some systems)
# - opencv-python: may require additional codecs for some image formats
# - dbr: requires Dynamsoft Barcode Reader license (trial available)
# - qreader: deep learning-based QR reader, may require additional ML dependencies

# System requirements:
# - Python 3.7+
# - Operating System: Windows, macOS, or Linux
# - Memory: 4GB+ RAM recommended for large datasets
# - Storage: 1GB+ free space for models and results
```

> If `pyzbar` complains about ZBar assertions, the code already wraps/stubs stderr and retries with gentle smoothing. fileciteturn0file3

---

## Configuration (YAML)

Create `configs/default.yaml` to control steps and budgets:

```yaml
# Per-image wall-clock budget (ms)
time_budget_ms: 300

# Steps will be executed in this **fixed order** if their gate fires:
# direct → threshold → rot_try → deskew → upsample → (optional fallback)
steps:
  - { name: threshold }             # adaptive local binarization
  - { name: rot_try, angles: [-45, -30, -10, 0, 10, 30, 45] }
  - { name: deskew }                # light affine deskew
  - { name: upsample, scale: 1.5 }  # bicubic upsample for small targets

# Optional ZXing/Pyzbar fallback when budget remains
fallback_zxing: false
```

Cues (contrast/tilt/size) are computed once per image and used to **gate** each step if it’s both *indicated* and *affordable within the remaining budget*. fileciteturn0file1

---

## Quickstart (CLI)

Batch‑run on a folder of images and save JSON results (with stage breakdowns and timings):

```bash
python run_folder_ours.py \
  --img_dir path/to/images \
  --cfg configs/default.yaml \
  --out out/results.json \
  --time_budget_ms 300 \
  -v
```

- The runner loads YAML, lists images, builds the default decoder backend (ZXing), runs the **retry pipeline**, and writes a JSON with metadata: total images, success count, avg time, and stage breakdowns. fileciteturn0file2  
- Core decode stages and time‑budget checks are implemented in `decode/retry.py`. fileciteturn0file1

**Example result entry** (per image):

```json
{
  "file": "IMG_0123.jpg",
  "ok": true,
  "texts": ["https://example.com"],
  "stage": "rot_try",
  "ms": 142.7,
  "time_budget_ms": 300
}
```

## Benchmarks & Protocol (summary)

- Datasets: three sets spanning in‑the‑wild and curated conditions; includes a difficult real‑world set with **low contrast / blur / small target / tilt** stress factors.  
- Protocol: sweep budgets **T ∈ {200, 300, 400, 500} ms**; exclude disk I/O; timeout counts as failure; log stage‑of‑success for curves.  
- Outcomes: TAPER improves success over strong open baselines by **2–7% overall** and up to **30% on stress subsets**, with a clear **knee** in the 200–500 ms range and **tighter tail latency**. See the PDF for tables/curves. fileciteturn0file0

---

## Troubleshooting

- **No codes detected**: enable `rot_try` with a wider angle set; consider `threshold` first (cheap) and `upsample` for tiny codes. fileciteturn0file1  
- **PyZbar crashes / assertions**: handled by safe wrappers and stderr redirection in the backend; update ZBar or switch to ZXing. fileciteturn0file3  
- **Tight budgets (200–300 ms)**: rotation attempts usually dominate the gains; as budget increases, deskew/upsample contribute more. fileciteturn0file0  
- **Windows/macOS install**: prefer `zxing-cpp` wheel; `pyzbar` is optional.

---

