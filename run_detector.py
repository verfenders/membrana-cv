#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import YOLODetector

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detector')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        headless=args.headless
    )
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()