#!/bin/bash

# Activate virtual environment
source ~/yolo_env/bin/activate

# Run detector
python run_detector.py --model ~/models/best.pt --headless