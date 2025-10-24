#!/usr/bin/env python3
import argparse
import sys
import os
import time

# Complex solution for PyTorch 2.6 on Windows
os.environ['TORCH_LOAD_DISABLE_SAFE_GLOBALS'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import YOLODetector

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detector')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Headless mode: {args.headless}")
    
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        headless=args.headless
    )
    
    try:
        print("Starting detection...")
        detector.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Cleaning up...")
        detector.cleanup()

if __name__ == "__main__":
    main()