(The file `/Users/aadarsh/Documents/AD coding/open cv/fire detection with yolo/README.md` exists, but contains only whitespace)
# YOLOv8 Fire & Smoke Detection

This repository contains code and models for detecting fire (and smoke) using YOLOv8 and OpenCV. It provides a lightweight demo script to run real-time detection from a webcam or video file and includes a pretrained model `fire.pt`.

## Contents
- `fire.py` — Main demo script to run detection (webcam / video / image).
- `fire.pt` — Pretrained YOLOv8 weights for fire detection.
- `fire-and-smoke-detection-yolov8-main/` — Auxiliary project folder (third-party code or experiments).
- `YOLOv8-Fire-and-Smoke-Detection-main/` — Another auxiliary folder containing example code or training scripts.

## Requirements
- Python 3.8+
- Recommended packages (install with pip):
	- `ultralytics` (YOLOv8)
	- `torch` (appropriate version for your GPU/OS)
	- `opencv-python`
	- `numpy`
	- `matplotlib` (optional, for visualizations)
	- `playsound` or use macOS `afplay` for alert sounds

You can create a virtual environment and install packages like this:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Example installs — adjust versions for your environment
pip install ultralytics torch opencv-python numpy matplotlib playsound
```

Note: On macOS `playsound` can fail with `ModuleNotFoundError: AppKit` if PyObjC is missing. A simple fallback is to use the built-in `afplay` command for local alerts.

## Usage

Run the demo with the pretrained weights included:

```bash
python fire.py --weights fire.pt --source 0
```

Options:
- `--weights` : path to model weights (default: `fire.pt`).
- `--source`  : `0` for webcam, path to video file, or path to an image.

Examples:
- Run webcam: `python fire.py --weights fire.pt --source 0`
- Run on video: `python fire.py --weights fire.pt --source videos/test.mp4`

## Training (overview)
If you want to retrain or fine-tune the model, use the YOLOv8 training workflow from `ultralytics`:

1. Prepare your dataset in YOLO format (images + labels).
2. Create a `.yaml` data config pointing to train/val sets and class names.
3. Run training:

```bash
from ultralytics import YOLO
# Example: train a fresh model
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100, imgsz=640)
```

Refer to the folders `fire-and-smoke-detection-yolov8-main/` and `YOLOv8-Fire-and-Smoke-Detection-main/` for example training scripts and dataset organization if present.

## Notes & Troubleshooting
- If detections are noisy, try increasing image size (`imgsz`) or training longer.
- Verify your `torch` installation matches your CUDA (or CPU) setup.
- For macOS sound alerts: if `playsound` fails, use `afplay /path/to/sound.wav` instead.

## License & Credits
This repository appears to combine example code and pretrained models. Verify licensing of the model files and any third-party folders before using in production.

## Next steps
- Add a `requirements.txt` or `environment.yml` for reproducible installs.
- (Optional) Create small scripts to evaluate model performance on a labeled validation set.

If you want, I can add a `requirements.txt` and a short example command to run tests or evaluate the model.

