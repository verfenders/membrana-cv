#!/usr/bin/env python3
"""
YOLO Object Detector with Basler Camera
Автор: [Ваше имя]
"""

import json
import time
import logging
from loguru import logger
import os
import sys

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, headless=False):
        """
        Инициализация YOLO детектора
        
        Args:
            model_path: путь к модели YOLO (.pt файл)
            conf_threshold: порог уверенности для детекции
            headless: режим без GUI
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.headless = headless
        self.model = None
        self.camera = None
        self.converter = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Настройка логирования
        self.setup_logging()
        
        # Загрузка модели
        self.load_model()
        
        # Инициализация камеры
        camera_initialized = self.setup_camera()
        
        if not camera_initialized:
            logger.error("❌ Camera initialization failed completely")
        elif self.camera is None:
            logger.info("📷 Running in simulation mode (no camera available)")
        else:
            logger.info("📷 Camera initialized successfully")
    
    def setup_logging(self):
        """Настройка логирования"""
        # Удаляем стандартный обработчик и добавляем свой
        logger.remove()
        logger.add(
            "detector.log",
            rotation="10 MB",
            retention="10 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
        )
        logger.info("🚀 YOLO Detector initialized")
    
    def load_model(self):
        """Загрузка модели YOLO с поддержкой PyTorch 2.6+"""
        try:
            from ultralytics import YOLO
            import torch
            import torch.serialization
            
            logger.info(f"🔄 Loading model: {self.model_path}")
            
            # Проверяем существование файла
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Попытка загрузки с безопасными глобалами для PyTorch 2.6+
            try:
                # Добавляем необходимые классы в безопасные глобалы
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
                
                self.model = YOLO(self.model_path)
                logger.info("✅ Model loaded successfully with safe_globals")
                
            except Exception as e1:
                logger.warning(f"⚠️ Safe globals method failed: {e1}")
                
                # Fallback: отключаем безопасную загрузку
                try:
                    import os
                    os.environ['TORCH_LOAD_DISABLE_SAFE_GLOBALS'] = '1'
                    self.model = YOLO(self.model_path)
                    logger.info("✅ Model loaded with disabled safe globals")
                    
                except Exception as e2:
                    logger.error(f"❌ All loading methods failed: {e2}")
                    
                    # Финальный fallback: используем официальную модель
                    logger.info("🔄 Falling back to official YOLOv8 model")
                    self.model = YOLO('yolov8n.pt')
                    logger.info("✅ Official YOLOv8 model loaded")
            
            logger.info(f"📊 Model info: {len(self.model.names)} classes")
            logger.info(f"📋 Classes: {list(self.model.names.values())}")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def setup_camera(self):
        """Настройка камеры Basler с обработкой ошибок"""
        try:
            from pypylon import pylon
            
            logger.info("🔄 Initializing camera...")
            
            # Ищем камеры
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                logger.error("❌ No Basler cameras found")
                logger.info("🔄 Running in simulation mode (no camera)")
                self.camera = None
                return True  # Продолжаем в режиме симуляции
            
            logger.info(f"✅ Found {len(devices)} camera(s):")
            for i, device in enumerate(devices):
                logger.info(f"  {i+1}. {device.GetModelName()} (SN: {device.GetSerialNumber()})")
            
            # Создаем и настраиваем камеру
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()
            
            logger.info(f"📷 Connected to: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # Базовые настройки камеры
            try:
                # Выключаем авторежимы
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.GainAuto.SetValue("Off")
                
                # Устанавливаем значения
                self.camera.ExposureTimeRaw.SetValue(30000)  # 30ms
                self.camera.GainRaw.SetValue(250)
                
                logger.info("✅ Camera settings applied")
                
            except Exception as config_error:
                logger.warning(f"⚠️ Some camera settings failed: {config_error}")
                # Продолжаем даже если настройки не применились
            
            # Настройка конвертера
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            # Запуск захвата
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            logger.info("✅ Camera setup completed and grabbing started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera setup failed: {e}")
            logger.info("🔄 Continuing in simulation mode")
            self.camera = None
            return True  # Продолжаем в режиме симуляции
    
    def capture_frame(self):
        """Захват кадра с камеры или симуляция"""
        if self.camera is None:
            logger.debug("📷 Camera is None - using simulation mode")
            return self._get_test_image()
        
        try:
            # Проверяем что камера активна
            if not self.camera.IsGrabbing():
                logger.warning("🔄 Camera was not grabbing, restarting...")
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                time.sleep(0.1)  # Даем время на запуск
            
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                frame = image.GetArray()
                grab_result.Release()
                self.frame_count += 1
                return frame
            else:
                logger.warning("❌ Frame grab failed")
                grab_result.Release()
                return None
                
        except Exception as e:
            logger.error(f"❌ Error capturing frame: {e}")
            
            # Пытаемся перезапустить камеру
            try:
                logger.info("🔄 Attempting to restart camera...")
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            except Exception as restart_error:
                logger.error(f"❌ Failed to restart camera: {restart_error}")
            
            return None
    
    def _get_test_image(self):
        """Создает тестовое изображение для симуляции"""
        import numpy as np
        import cv2
        import random
        
        # Создаем тестовое изображение 640x480
        height, width = 480, 640
        test_image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
        
        # Добавляем "объекты" - случайные прямоугольники
        num_objects = random.randint(1, 5)
        for i in range(num_objects):
            x1 = random.randint(50, width - 100)
            y1 = random.randint(50, height - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, 2)
            
            # Подпись объекта
            cv2.putText(test_image, f"Object_{i}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Информация о режиме
        cv2.putText(test_image, "SIMULATION MODE - NO CAMERA", (50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(test_image, f"Objects: {num_objects}", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(test_image, f"Frame: {self.frame_count}", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        self.frame_count += 1
        return test_image
    
    def detect(self):
        """Выполнение детекции на одном кадре"""
        # Захват кадра
        frame = self.capture_frame()
        
        if frame is None:
            return {
                "error": "Failed to capture frame",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "objects_detected": 0,
                "objects": [],
                "camera_status": "error"
            }
        
        # Детекция
        start_time = time.time()
        try:
            results = self.model(frame, verbose=False)
            inference_time = (time.time() - start_time) * 1000
            
            detection_data = self.process_results(results, inference_time)
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            detection_data = {
                "error": f"Detection failed: {e}",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "objects_detected": 0,
                "objects": [],
                "inference_time_ms": 0
            }
            inference_time = 0
        
        # Мониторинг ресурсов
        import psutil
        detection_data.update({
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "camera_status": "active" if self.camera else "simulation",
            "frame_count": self.frame_count,
            "uptime_seconds": round(time.time() - self.start_time, 2)
        })
        
        if "error" not in detection_data:
            logger.info(f"✅ Detection: {detection_data['objects_detected']} objects, {detection_data['inference_time_ms']:.1f}ms")
        else:
            logger.warning(f"⚠️ Detection with error: {detection_data['error']}")
        
        return detection_data
    
    def process_results(self, results, inference_time):
        """Обработка результатов детекции YOLO"""
        detection_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "inference_time_ms": round(inference_time, 2),
            "objects_detected": 0,
            "objects": []
        }
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                detection_data["objects_detected"] = len(boxes)
                
                for i, box in enumerate(boxes):
                    conf = box.conf[0].cpu().numpy()
                    if conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        object_info = {
                            "id": i,
                            "class_id": cls,
                            "class_name": self.model.names[cls],
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        }
                        detection_data["objects"].append(object_info)
        
        return detection_data
    
    def save_detection_result(self, detection_data, output_dir="detections"):
        """Сохранение результатов детекции в JSON файл"""
        try:
            # Создаем папку если не существует
            os.makedirs(output_dir, exist_ok=True)
            
            # Имя файла с временной меткой
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{output_dir}/detection_{timestamp}_{self.frame_count:06d}.json"
            
            # Сохраняем в JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Results saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return None
    
    def get_camera_status(self):
        """Возвращает статус камеры"""
        if self.camera is None:
            return "simulation"
        elif self.camera.IsGrabbing():
            return "active"
        else:
            return "inactive"
    
    def print_status(self):
        """Печатает статус системы"""
        status = self.get_camera_status()
        fps = self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0
        
        logger.info("📊 System Status:")
        logger.info(f"  - Camera: {status}")
        logger.info(f"  - Model: {'loaded' if self.model else 'not loaded'}")
        logger.info(f"  - Frames processed: {self.frame_count}")
        logger.info(f"  - FPS: {fps:.1f}")
        logger.info(f"  - Uptime: {time.time() - self.start_time:.1f}s")
        logger.info(f"  - Mode: {'headless' if self.headless else 'GUI'}")
    
    def run(self, save_results=False, output_dir="detections", interval=2.0):
        """
        Основной цикл детекции
        
        Args:
            save_results: сохранять ли результаты в файлы
            output_dir: папка для сохранения результатов
            interval: интервал между детекциями в секундах
        """
        logger.info("🚀 Starting detection loop...")
        logger.info(f"📁 Results will be saved to: {output_dir}" if save_results else "📁 Results will not be saved")
        logger.info(f"⏱️  Detection interval: {interval}s")
        
        self.print_status()
        
        try:
            while True:
                # Выполняем детекцию
                result = self.detect()
                
                # Выводим результат в консоль
                print(json.dumps(result, indent=2))
                
                # Сохраняем в файл если нужно
                if save_results and "error" not in result:
                    self.save_detection_result(result, output_dir)
                
                # Выводим статус каждые 10 кадров
                if self.frame_count % 10 == 0:
                    self.print_status()
                
                # Пауза между детекциями
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Detection stopped by user")
        except Exception as e:
            logger.error(f"💥 Unexpected error in detection loop: {e}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            if self.camera is not None:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                logger.info("✅ Camera resources released")
            else:
                logger.info("✅ Simulation mode - no camera to cleanup")
        except Exception as e:
            logger.error(f"❌ Error during camera cleanup: {e}")
        
        # Статистика
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        logger.info("📈 Final Statistics:")
        logger.info(f"  - Total frames: {self.frame_count}")
        logger.info(f"  - Total time: {total_time:.1f}s")
        logger.info(f"  - Average FPS: {fps:.1f}")
        logger.info(f"  - Camera status: {self.get_camera_status()}")
        
        logger.info("✅ All resources cleaned up")


# Пример использования
if __name__ == "__main__":
    # Тестовый запуск
    detector = YOLODetector(
        model_path="models/best.pt",
        conf_threshold=0.5,
        headless=True
    )
    
    try:
        detector.run(save_results=True, interval=2.0)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        detector.cleanup()