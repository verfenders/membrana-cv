
**`scripts/setup_server.sh`**:
```bash
#!/bin/bash

echo "🐍 Setting up YOLO detector on Ubuntu server..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Install system dependencies
sudo apt install -y \
    build-essential \
    libpcap-dev \
    libusb-1.0-0-dev \
    libgl1 \
    libglib2.0-0

# Setup camera permissions
sudo usermod -a -G video $USER
sudo usermod -a -G plugdev $USER

# Create virtual environment
python3.12 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "🚀 Activate environment: source ~/yolo_env/bin/activate"
echo "🎯 Run detector: python run_detector.py --model path/to/model.pt --headless"