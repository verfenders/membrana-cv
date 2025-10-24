# YOLO Object Detector with Basler Camera

Real-time object detection system using YOLO and Basler cameras.

## Installation
```bash
git clone https://github.com/verfenders/membrana-cv
cd yolo-detector
pip install -r requirements.txt

python run_detector.py --model models/best.pt --headless

export TORCH_LOAD_DISABLE_SAFE_GLOBALS=1