import os

class Config:
    # Model settings
    YOLO_MODEL = "yolov8x.pt"
    
    # Advanced VLM settings
    VLM_MODEL = "llava-hf/llava-1.5-7b-hf"  # Or "Salesforce/blip2-opt-2.7b"
    USE_ADVANCED_VLM = True
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Scene understanding settings
    ENABLE_SCENE_UNDERSTANDING = True
    ENABLE_RELATIONSHIP_DETECTION = True
    ENABLE_ACTIVITY_RECOGNITION = True
    MAX_SCENE_OBJECTS = 20
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    
    # Display settings
    WINDOW_NAME = "ORION ADVANCED VLM"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    FPS_UPDATE_RATE = 10
    
    # Speech settings
    SPEECH_TIMEOUT = 8
    ENERGY_THRESHOLD = 1000
    PAUSE_THRESHOLD = 0.8
    WAKE_WORD = "orion"
    
    # Description settings
    MAX_DESCRIPTION_LENGTH = 200
    DESCRIPTION_UPDATE_RATE = 2
    
    # Performance settings
    DETECTION_INTERVAL = 0.033
    INFERENCE_SIZE = 640
    VLM_PROCESSING_INTERVAL = 2.0
    
    # Font constants
    LABEL_FONT = "FONT_HERSHEY_SIMPLEX"
    LABEL_FONT_SCALE = 0.6
    LABEL_THICKNESS = 2
    LABEL_MARGIN = 15
    DESCRIPTION_FONT_SCALE = 0.5
    
    # Advanced UI colors
    HOLO_COLORS = {
        'human_animal': (255, 255, 0),    # Yellow for humans & animals
        'vehicle': (255, 0, 0),           # Red for vehicles
        'electronic': (0, 255, 0),        # Green for electronics
        'relationship': (255, 0, 255),    # Magenta for relationships
        'activity': (0, 255, 255),        # Cyan for activities
        'vlm_active': (255, 165, 0),      # Orange for VLM status
        'scene_context': (255, 255, 255), # White for scene info
        'classroom': (0, 255, 255),       # Cyan for classroom objects
        'food': (255, 165, 0),            # Orange for food items
        'default': (128, 0, 128),         # Purple for other objects
        'text_glow': (0, 255, 255),       # Cyan for text glow
        'background': (0, 0, 0),          # Black
        'sky_blue': (255, 191, 0),        # Sky blue for human detection
        'recording_overlay': (100, 100, 50),  # Sky blue overlay color
        'attribute_highlight': (255, 255, 0)  # Yellow for attribute highlights
    }
    
    # Animation settings
    BASE_CIRCLE_RADIUS = 25
    CIRCLE_THICKNESS = 2
    OUTER_CIRCLE_THICKNESS = 4
    ROTATION_SPEED = 2
    PULSE_SPEED = 0.1
    
    # Enhanced category mappings
    HUMAN_ANIMAL_CLASSES = [
        'person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'panda', 'lion', 'tiger'
    ]
    
    VEHICLE_CLASSES = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 
        'boat', 'airplane', 'helicopter', 'ship'
    ]
    
    CLASSROOM_OBJECTS = [
        'book', 'laptop', 'cell phone', 'bottle', 'cup', 'chair', 'dining table',
        'tv', 'mouse', 'keyboard', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'wine glass', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'potted plant'
    ]
    
    # Attribute detection settings
    COLOR_DETECTION_ENABLED = True
    SIZE_CLASSIFICATION_ENABLED = True
    POSITION_DETECTION_ENABLED = True
    MIN_ATTRIBUTE_CONFIDENCE = 0.3

    # Activity recognition patterns
    ACTIVITY_PATTERNS = {
        'sitting': ['chair', 'sofa', 'bench'],
        'using_phone': ['cell phone', 'person'],
        'reading': ['book', 'person', 'newspaper'],
        'drinking': ['bottle', 'cup', 'person'],
        'working': ['laptop', 'person', 'desk'],
        'carrying': ['person', 'backpack', 'handbag', 'suitcase'],
        'watching': ['tv', 'person', 'monitor'],
        'eating': ['person', 'apple', 'sandwich', 'pizza', 'banana']
    }

    # Question patterns for natural language understanding
    QUESTION_PATTERNS = {
        'what': ['what is', 'what are', 'what does', 'what do'],
        'where': ['where is', 'where are'],
        'who': ['who is', 'who are'],
        'how': ['how many', 'how much'],
        'why': ['why is', 'why are'],
        'activity': ['what is doing', 'what are they doing', 'what is happening'],
        'relationship': ['carrying', 'holding', 'has', 'with', 'using']
    }

    def __init__(self):
        print("🔧 Advanced VLM Configuration Loaded")
        
        # Validate VLM settings
        if self.USE_ADVANCED_VLM:
            if self.OPENAI_API_KEY:
                print("✅ Advanced VLM: GPT-4 Vision available")
            else:
                print("🔄 Advanced VLM: Using open-source models (LLaVA/BLIP-2)")
        else:
            print("⚠️  Advanced VLM: Disabled - Using basic object detection")
        
        if self.ENABLE_SCENE_UNDERSTANDING:
            print("🎯 Scene Understanding: ENABLED")
        if self.ENABLE_RELATIONSHIP_DETECTION:
            print("🔗 Relationship Detection: ENABLED") 
        if self.ENABLE_ACTIVITY_RECOGNITION:
            print("🏃 Activity Recognition: ENABLED")