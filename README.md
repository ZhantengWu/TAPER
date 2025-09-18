# QR Code Robust Reading Project

A step-by-step retry strategy for robust QR code and barcode reading with time budget control. This project implements multiple decoding approaches and provides comprehensive evaluation metrics for ICASSP paper submission.

## Overview

This project compares different QR code/barcode reading strategies:
- **Baseline A**: Direct pyzbar decoding
- **Baseline C**: Upsampling + pyzbar decoding  
- **Our Method**: Step-by-step retry strategy with early stopping and time budget control

## Features

- **Multi-stage retry strategy**: Direct → Upsample → Threshold → Deskew → Rotation attempts
- **Time budget control**: Configurable maximum processing time per image
- **Early stopping**: Stop as soon as successful decode is achieved
- **Comprehensive evaluation**: Success rate, misread rate, processing speed metrics
- **Configurable pipeline**: YAML-based configuration for easy experimentation
- **Multiple backends**: Support for pyzbar and ZXing decoders

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n qrrobust python=3.10 -y
conda activate qrrobust

# Install dependencies
pip install opencv-python pyzbar pillow numpy scikit-image matplotlib pandas tqdm pyyaml seaborn
```

### Optional: ZXing Support

For ZXing fallback support, download the ZXing command line jar:
```bash
# Download ZXing jar (optional)
wget https://repo1.maven.org/maven2/com/google/zxing/javase/3.5.1/javase-3.5.1.jar -O zxing.jar
```

## Project Structure

```
qrrobust/
├── data/                      # Test datasets
├── results/                   # Output results (JSON/CSV/plots)
├── configs/
│   └── retry_simple.yaml      # Configuration file
├── src/
│   ├── utils/
│   │   ├── io.py              # File I/O utilities
│   │   └── timer.py           # Timing utilities
│   ├── preprocess/
│   │   └── ops.py             # Image preprocessing operations
│   ├── decode/
│   │   ├── backend.py         # Decoder backends (pyzbar/ZXing)
│   │   ├── retry.py           # Core retry strategy
│   │   ├── run_folder_baseA.py# Baseline A runner
│   │   ├── run_folder_baseC.py# Baseline C runner
│   │   └── run_folder_ours.py # Our method runner
│   └── eval/
│       ├── metrics.py         # Evaluation metrics
│       └── summarize.py       # Results comparison
└── README.md
```

## Quick Start

### 1. Prepare Test Data

Place your QR code/barcode images in a directory (e.g., `data/test_images/`):

```bash
mkdir -p data/test_images
# Copy your test images to data/test_images/
```

### 2. Run Methods

```bash
# Baseline A: Direct pyzbar decoding
python src/decode/run_folder_baseA.py --img_dir data/test_images --out results/baseA.json

# Baseline C: Upsampling + pyzbar
python src/decode/run_folder_baseC.py --img_dir data/test_images --out results/baseC.json

# Our method: Step-by-step retry strategy
python src/decode/run_folder_ours.py --img_dir data/test_images --cfg configs/retry_simple.yaml --out results/ours.json
```

### 3. Evaluate Results

```bash
# Individual method evaluation
python src/eval/metrics.py --json results/baseA.json --method_name "Baseline A"
python src/eval/metrics.py --json results/baseC.json --method_name "Baseline C"
python src/eval/metrics.py --json results/ours.json --method_name "Our Method"

# Compare all methods
python src/eval/summarize.py \
  --items results/baseA.json:BaselineA,results/baseC.json:BaselineC,results/ours.json:OurMethod \
  --out results/comparison.csv
```

## Configuration

Edit `configs/retry_simple.yaml` to customize the retry strategy:

```yaml
time_budget_ms: 120           # Maximum time per image (ms)
steps:
  - name: direct              # Direct decoding
  - name: upsample           # Upsampling
    scale: 1.5               # Scale factor
  - name: threshold          # Adaptive thresholding
  - name: deskew            # Light deskewing
  - name: rot_try           # Rotation attempts
    angles: [-10, -5, 5, 10] # Rotation angles (degrees)
fallback_zxing: false        # Enable ZXing fallback
```

## Evaluation Metrics

The project calculates the following metrics:

- **Decode Success Rate (%)**: Percentage of images successfully decoded
- **Misread Rate (%)**: Percentage of incorrect decodes (requires ground truth)
- **Average Time (ms)**: Average processing time per image
- **FPS**: Processing speed (frames per second)
- **Stage Breakdown**: Success rate at each processing stage

## Advanced Usage

### Custom Preprocessing

Modify `src/preprocess/ops.py` to add custom preprocessing operations:

```python
def custom_enhancement(img):
    # Your custom preprocessing
    return enhanced_img
```

### Ground Truth Evaluation

For misread rate calculation, provide ground truth files:

```bash
# Create ground truth directory
mkdir -p data/ground_truth

# Create .txt files with same names as images
echo "Expected QR content" > data/ground_truth/image1.txt

# Run evaluation with ground truth
python src/eval/metrics.py --json results/ours.json --gt_dir data/ground_truth
```

### Batch Processing

Process multiple datasets:

```bash
# Process multiple directories
for dataset in dataset1 dataset2 dataset3; do
    python src/decode/run_folder_ours.py \
        --img_dir data/$dataset \
        --cfg configs/retry_simple.yaml \
        --out results/${dataset}_ours.json
done
```

### Ablation Studies

Create different configurations for ablation studies:

```bash
# Copy base config
cp configs/retry_simple.yaml configs/no_rotation.yaml

# Edit no_rotation.yaml to remove rot_try step
# Then run:
python src/decode/run_folder_ours.py \
    --img_dir data/test_images \
    --cfg configs/no_rotation.yaml \
    --out results/no_rotation.json
```

## Output Format

### JSON Results Format

Each method outputs results in JSON format:

```json
[
  {
    "file": "image1.jpg",
    "ok": true,
    "texts": ["QR_CODE_CONTENT"],
    "stage": "upsample",
    "ms": 45.2
  },
  ...
]
```

### CSV Summary Format

The comparison tool generates CSV tables:

| Method | Total Images | Successful Decodes | Decode Success Rate (%) | Avg Time (ms) | FPS |
|--------|-------------|-------------------|------------------------|---------------|-----|
| BaselineA | 100 | 75 | 75.00 | 12.3 | 81.3 |
| BaselineC | 100 | 82 | 82.00 | 18.7 | 53.5 |
| OurMethod | 100 | 89 | 89.00 | 35.4 | 28.2 |

## Troubleshooting

### Common Issues

1. **pyzbar not working**: Install system dependencies
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libzbar0
   
   # macOS
   brew install zbar
   
   # Windows: Download from https://github.com/NuGet/Home/issues/4301
   ```

2. **Memory issues with large datasets**: Process in batches or reduce image sizes

3. **ZXing not found**: Ensure Java is installed and zxing.jar is in the correct path

### Performance Tips

- Resize large images to max 640px for faster processing
- Use `--verbose` flag to monitor progress
- Adjust `time_budget_ms` based on your speed/accuracy requirements

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{qrrobust2024,
  title={Robust QR Code Reading with Step-by-Step Retry Strategy},
  author={Your Name},
  booktitle={ICASSP 2024},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
