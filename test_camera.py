#!/usr/bin/env python3
"""
Диагностика подключения камеры Basler
"""

import subprocess
import os

def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def diagnose_camera():
    print("🔧 Camera Diagnostics")
    print("=" * 60)
    
    # 1. Проверка USB устройств
    print("1. 📷 Checking USB devices:")
    success, stdout, stderr = run_command("lsusb")
    if success:
        print(stdout)
        if "Basler" in stdout or "1394" in stdout:
            print("✅ Basler camera found via USB")
        else:
            print("❌ No Basler camera found in USB devices")
    else:
        print("❌ lsusb failed")
    
    # 2. Проверка видео устройств
    print("\n2. 📹 Checking video devices:")
    success, stdout, stderr = run_command("ls -la /dev/video* 2>/dev/null || echo 'No video devices'")
    print(stdout)
    
    # 3. Проверка IEEE 1394 (FireWire) для Basler
    print("\n3. 🔥 Checking FireWire devices:")
    success, stdout, stderr = run_command("ls -la /dev/fw* 2>/dev/null || echo 'No FireWire devices'")
    print(stdout)
    
    # 4. Проверка групп пользователя
    print("\n4. 👤 Checking user groups:")
    success, stdout, stderr = run_command("groups")
    print(f"User groups: {stdout}")
    
    # 5. Проверка через pypylon
    print("\n5. 🐍 Checking via pypylon:")
    try:
        from pypylon import pylon
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        print(f"Pypylon found {len(devices)} devices:")
        for i, device in enumerate(devices):
            print(f"  {i+1}. {device.GetModelName()} (SN: {device.GetSerialNumber()})")
    except Exception as e:
        print(f"❌ Pypylon error: {e}")
    
    # 6. Проверка установленных пакетов
    print("\n6. 📦 Checking installed packages:")
    success, stdout, stderr = run_command("dpkg -l | grep -E '(pylon|basler)' || echo 'No Basler packages found'")
    print(stdout)
    
    print("=" * 60)

if __name__ == "__main__":
    diagnose_camera()