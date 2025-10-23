import json
import time
import logging
from loguru import logger

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, headless=False):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.headless = headless
        self.model = None
        self.camera = None
        
        self.setup_logging()
        self.load_model()
        if not self.headless:
            self.setup_camera()
    
    def setup_logging(self):
        logger.add("detector.log", rotation="10 MB", level="INFO")
        logger.info("YOLO Detector initialized")
    
    def load_model(self):
        from ultralytics import YOLO
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("Model loaded successfully")
    
    def setup_camera(self):
        try:
            from pypylon import pylon
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            
            # Настройки камеры
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.GainAuto.SetValue("Off")
            self.camera.ExposureTimeRaw.SetValue(30000)
            self.camera.GainRaw.SetValue(250)
            
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            logger.info("Camera setup completed")
            
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            raise
    
    def capture_frame(self):
        try:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                frame = image.GetArray()
                grab_result.Release()
                return frame
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
        return None
    
    def detect(self):
        frame = self.capture_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}
        
        start_time = time.time()
        results = self.model(frame, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        detection_data = self.process_results(results, inference_time)
        
        import psutil
        detection_data.update({
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent
        })
        
        logger.info(f"Detection completed: {detection_data['objects_detected']} objects")
        return detection_data
    
    def process_results(self, results, inference_time):
        detection_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "inference_time_ms": round(inference_time, 2),
            "objects_detected": 0,
            "objects": []
        }
        
        for result in results:
            if result.boxes is not None:
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
                                "x1": float(x1), "y1": float(y1),
                                "x2": float(x2), "y2": float(y2)
                            }
                        }
                        detection_data["objects"].append(object_info)
        
        return detection_data
    
    def run(self):
        logger.info("Starting detection loop...")
        try:
            while True:
                result = self.detect()
                print(json.dumps(result, indent=2))
                
                # Сохранение в файл
                with open(f"detection_{int(time.time())}.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("Detection stopped")
    
    def cleanup(self):
        if self.camera:
            self.camera.StopGrabbing()
            self.camera.Close()
        logger.info("Resources cleaned up")