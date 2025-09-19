# TAPER: Time‑Aware Pipeline for Efficient, Robust QR/Barcode Decoding

> Training‑free, **budgeted** decoding that adapts compute to image difficulty and **stops at first success**.

TAPER turns a standard QR/Barcode reader into an **anytime system**: it uses lightweight cues (contrast / tilt / target size) to **conditionally** trigger enhancement steps (thresholding, rotation, deskew, upsampling), attempts a decode after each step, and **respects a per‑image time budget**. Accuracy increases with budget while easy cases exit early.

- Paper / method overview: see the included PDF. fileciteturn0file0  
- Core retry algorithm (time‑aware conditional pipeline): `decode/retry.py`. fileciteturn0file1  
- Backends (ZXing, Pyzbar, QReader, Dynamsoft): `decode/backend.py`. fileciteturn0file3  
- CLI runner for batches: `run_folder_ours.py`. fileciteturn0file2

---

## ✨ Key Features

- **Budgeted decoding**: wall‑clock time budget per image (e.g., 200–500 ms).  
- **Early‑stop anytime behavior**: return as soon as a backend succeeds.  
- **Lightweight cues**: fast, on‑device estimates of contrast, tilt/skew, and target size.  
- **Conditional steps**: trigger only the fixes that are likely to help (threshold / rotate / deskew / upsample).  
- **Pluggable backends**: ZXing‑CPP (default), Pyzbar/ZBar, QReader, Dynamsoft DBR.  
- **Reproducible evaluation**: logs per‑stage timings, success/timeout, and stage‑of‑success for tables and curves.

See the PDF for the full method, ablations, and budgeted evaluation protocol. fileciteturn0file0

---

## 🧱 Repository Layout (reference)

```
taper/
├─ README.md
├─ Wu.pdf                        # Paper/extended abstract (method + experiments)
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

## 📦 Installation

Tested on Python 3.9–3.11.

```bash
# 1) (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install core dependencies
pip install -U opencv-python numpy pyyaml

# 3) Install at least one backend (ZXing is the default)
pip install zxing-cpp

# (Optional) Additional backends
pip install pyzbar              # requires system zbar on Linux; on Windows/macOS wheels usually bundle it
pip install qreader             # optional, for QReaderBackend
pip install dbr                 # optional, Dynamsoft Barcode Reader
```

> If `pyzbar` complains about ZBar assertions, the code already wraps/stubs stderr and retries with gentle smoothing. fileciteturn0file3

---

## ⚙️ Configuration (YAML)

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

## 🚀 Quickstart (CLI)

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

---

## 🐍 Python API (embed in your app)

```python
import cv2
from decode.backend import get_default_backend
from decode.retry import retry_decode

img = cv2.imread("path/to/img.jpg")
backend = get_default_backend()  # ZXing by default  fileciteturn0file3

cfg = {
    "time_budget_ms": 300,
    "steps": [
        {"name": "threshold"},
        {"name": "rot_try", "angles": [-10, -5, 5, 10]},
        {"name": "deskew"},
        {"name": "upsample", "scale": 1.5},
    ],
    "fallback_zxing": False
}

texts, final_stage, total_ms = retry_decode(img, backend, cfg)  # fileciteturn0file1
print(texts, final_stage, total_ms)
```

---

## 📊 Benchmarks & Protocol (summary)

- Datasets: three sets spanning in‑the‑wild and curated conditions; includes a difficult real‑world set with **low contrast / blur / small target / tilt** stress factors.  
- Protocol: sweep budgets **T ∈ {200, 300, 400, 500} ms**; exclude disk I/O; timeout counts as failure; log stage‑of‑success for curves.  
- Outcomes: TAPER improves success over strong open baselines by **2–7% overall** and up to **30% on stress subsets**, with a clear **knee** in the 200–500 ms range and **tighter tail latency**. See the PDF for tables/curves. fileciteturn0file0

---

## 🔧 Troubleshooting

- **No codes detected**: enable `rot_try` with a wider angle set; consider `threshold` first (cheap) and `upsample` for tiny codes. fileciteturn0file1  
- **PyZbar crashes / assertions**: handled by safe wrappers and stderr redirection in the backend; update ZBar or switch to ZXing. fileciteturn0file3  
- **Tight budgets (200–300 ms)**: rotation attempts usually dominate the gains; as budget increases, deskew/upsample contribute more. fileciteturn0file0  
- **Windows/macOS install**: prefer `zxing-cpp` wheel; `pyzbar` is optional.

---

## 📁 Reproducibility & Logs

Every run records per‑image JSON with:
- success / fail / timeout,  
- **final stage** (direct / threshold / rot_try / deskew / upsample / zxing_fallback),  
- per‑image **milliseconds**, and global metadata (avg ms, total successes, stage histogram). fileciteturn0file2

These feed directly into tables, anytime curves, and latency CDFs in the paper. fileciteturn0file0

---

## 🧪 Extending Backends

Add your own backend by subclassing `DecoderBackend` and wiring it in `create_backend()`. ZXing is the default; fallbacks can be toggled in config. fileciteturn0file3

---

## 📚 Citation

If you use this project, please cite the paper included in `Wu.pdf`. fileciteturn0file0

```bibtex
@inproceedings{wu2025taper,
  title={TAPER: Time-Aware Pipeline for Efficient Robust QR/Barcode Decoding},
  author={Wu, Zhanteng and Diao, Hongyue and Wei, Hao},
  booktitle={ICASSP (under submission)},
  year={2025}
}
```

---

## 📝 License

Add your license here (e.g., MIT).

---

## 🙌 Acknowledgments

See the paper for funding and acknowledgments. fileciteturn0file0
