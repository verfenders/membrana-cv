def __init__(self, model_path, conf_threshold=0.5, headless=False):
    self.model_path = model_path
    self.conf_threshold = conf_threshold
    self.headless = headless
    self.model = None
    self.camera = None
    self.converter = None
    
    self.setup_logging()
    self.load_model()
    
    # Инициализируем камеру и проверяем результат
    camera_initialized = self.setup_camera()
    
    if not camera_initialized:
        logger.error("❌ Camera initialization failed completely")
    elif self.camera is None:
        logger.info("📷 Running in simulation mode (no camera available)")
    else:
        logger.info("📷 Camera initialized successfully")