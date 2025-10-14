from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional  # ADD Optional HERE
import time

class AdvancedObjectDetector:
    def __init__(self, config):
        self.config = config
        self.model = YOLO('yolov8m.pt')
        self.class_names = self.model.names
        
        # Activity recognition patterns
        self.activity_patterns = {
            'sitting': ['chair', 'sofa', 'bench'],
            'using_phone': ['cell phone', 'person'],
            'reading': ['book', 'person', 'newspaper'],
            'drinking': ['bottle', 'cup', 'person'],
            'working': ['laptop', 'person', 'desk'],
            'carrying': ['person', 'backpack', 'handbag', 'suitcase']
        }
        
        print("🚀 Advanced Object Detector with Relationship Analysis Ready")
    
    def detect_with_relationships(self, frame: np.ndarray) -> Dict:
        """Advanced detection with relationships and activities"""
        # Basic object detection
        detections = self.detect(frame)
        
        # Enhanced analysis
        relationships = self._detect_relationships(frame, detections)
        activities = self._detect_activities(detections, relationships)
        scene_context = self._analyze_scene_context(detections, relationships, activities)
        
        return {
            'objects': detections,
            'relationships': relationships,
            'activities': activities,
            'scene_context': scene_context,
            'timestamp': time.time()
        }
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Fast object detection"""
        try:
            results = self.model(
                frame, 
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=0.5,  # Add IOU threshold
                imgsz=320,  # Smaller image size for faster processing
                half=False,  # Use full precision on CPU
                verbose=False,
                max_det=10  # Limit detections
            )
            # ... rest of your code

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = self.class_names[class_id]

                        # Add unique ID for relationship tracking
                        obj_id = f"{label}_{i}_{time.time()}"

                        attributes = self._extract_attributes(frame, [x1, y1, x2, y2], label)

                        detections.append({
                            'id': obj_id,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'label': label,
                            'class_id': class_id,
                            'attributes': attributes,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        })

            return sorted(detections, key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _detect_relationships(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Detect spatial and functional relationships"""
        relationships = []
        
        persons = [d for d in detections if d['label'] == 'person']
        objects = [d for d in detections if d['label'] != 'person']
        
        # Person-object relationships
        for person in persons:
            for obj in objects:
                relationship = self._analyze_relationship(person, obj)
                if relationship:
                    relationships.append(relationship)
        
        # Object-object relationships
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                relationship = self._analyze_object_relationship(obj1, obj2)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _analyze_relationship(self, person: Dict, obj: Dict) -> Optional[Dict]:
        """Analyze relationship between person and object"""
        p_center = person['center']
        o_center = obj['center']
        
        distance = np.sqrt((p_center[0] - o_center[0])**2 + (p_center[1] - o_center[1])**2)
        
        # Relationship thresholds
        if distance < 150:  # Close proximity
            rel_type = self._determine_interaction_type(person, obj)
            return {
                'type': rel_type,
                'person': person,
                'object': obj,
                'distance': distance,
                'confidence': min(person['confidence'], obj['confidence'])
            }
        
        return None
    
    def _analyze_object_relationship(self, obj1: Dict, obj2: Dict) -> Optional[Dict]:
        """Analyze relationships between two objects"""
        o1_center = obj1['center']
        o2_center = obj2['center']
        
        distance = np.sqrt((o1_center[0] - o2_center[0])**2 + (o1_center[1] - o2_center[1])**2)
        
        # Objects that are often used together
        common_pairs = [
            ('laptop', 'mouse'),
            ('laptop', 'keyboard'),
            ('cup', 'bottle'),
            ('book', 'glasses')
        ]
        
        for pair in common_pairs:
            if (obj1['label'] in pair and obj2['label'] in pair):
                return {
                    'type': 'used_with',
                    'object1': obj1,
                    'object2': obj2,
                    'distance': distance,
                    'confidence': min(obj1['confidence'], obj2['confidence'])
                }
        
        return None
    
    def _determine_interaction_type(self, person: Dict, obj: Dict) -> str:
        """Determine type of interaction"""
        obj_label = obj['label'].lower()
        
        interaction_map = {
            'cell phone': 'using phone',
            'laptop': 'using computer', 
            'book': 'reading',
            'cup': 'drinking from',
            'bottle': 'drinking from',
            'chair': 'sitting on',
            'tv': 'watching',
            'backpack': 'carrying',
            'handbag': 'carrying',
            'mouse': 'using',
            'keyboard': 'using'
        }
        
        return interaction_map.get(obj_label, 'interacting with')
    
    def _detect_activities(self, detections: List[Dict], relationships: List[Dict]) -> List[Dict]:
        """Detect activities based on objects and relationships"""
        activities = []
        
        for person in [d for d in detections if d['label'] == 'person']:
            person_relationships = [r for r in relationships if r['person']['id'] == person['id']]
            
            # Determine activity based on relationships
            activity = self._infer_activity(person, person_relationships)
            if activity:
                activities.append(activity)
        
        return activities
    
    def _infer_activity(self, person: Dict, relationships: List[Dict]) -> Optional[Dict]:
        """Infer activity from relationships"""
        if not relationships:
            return None
            
        # Count relationship types
        rel_types = [r['type'] for r in relationships]
        
        activity_info = {
            'person': person,
            'primary_activity': self._determine_primary_activity(rel_types),
            'involved_objects': [r['object'] for r in relationships],
            'confidence': person['confidence']
        }
        
        return activity_info
    
    def _determine_primary_activity(self, rel_types: List[str]) -> str:
        """Determine primary activity from relationship types"""
        if 'using phone' in rel_types:
            return 'using mobile phone'
        elif 'reading' in rel_types:
            return 'reading'
        elif 'using computer' in rel_types:
            return 'working on computer'
        elif 'drinking from' in rel_types:
            return 'drinking'
        elif 'sitting on' in rel_types:
            return 'sitting'
        elif 'carrying' in rel_types:
            return 'carrying items'
        elif 'watching' in rel_types:
            return 'watching screen'
        else:
            return 'present in scene'
    
    def _analyze_scene_context(self, detections: List[Dict], relationships: List[Dict], activities: List[Dict]) -> Dict:
        """Analyze overall scene context"""
        object_counts = {}
        for det in detections:
            label = det['label']
            object_counts[label] = object_counts.get(label, 0) + 1
        
        return {
            'total_objects': len(detections),
            'object_counts': object_counts,
            'total_people': len([d for d in detections if d['label'] == 'person']),
            'total_activities': len(activities),
            'total_relationships': len(relationships),
            'primary_activities': [a['primary_activity'] for a in activities],
            'most_common_object': max(object_counts, key=object_counts.get) if object_counts else 'none'
        }
    
    def _extract_attributes(self, frame: np.ndarray, bbox: List[int], label: str) -> Dict:
        """Extract visual attributes"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        return {
            'color': self._detect_color(roi),
            'size': self._estimate_size(bbox),
            'position': self._get_position(bbox, frame.shape)
        }
    
    def _detect_color(self, roi: np.ndarray) -> str:
        """Detect dominant color"""
        if roi.size == 0:
            return "unknown"
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate average color
            avg_color = np.mean(roi, axis=(0, 1))
            
            # Simple color mapping
            colors = {
                'red': ([0, 120, 70], [10, 255, 255]),
                'blue': ([110, 50, 50], [130, 255, 255]),
                'green': ([36, 50, 70], [89, 255, 255]),
                'yellow': ([20, 100, 100], [30, 255, 255]),
                'white': ([0, 0, 200], [180, 55, 255]),
                'black': ([0, 0, 0], [180, 255, 50])
            }
            
            # Find closest color
            hsv_avg = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            for color_name, (lower, upper) in colors.items():
                if (lower[0] <= hsv_avg[0] <= upper[0] and
                    lower[1] <= hsv_avg[1] <= upper[1] and
                    lower[2] <= hsv_avg[2] <= upper[2]):
                    return color_name
            
            return "unknown"
        except:
            return "unknown"
    
    def _estimate_size(self, bbox: List[int]) -> str:
        """Estimate object size relative to frame"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area < 1000:
            return "small"
        elif area < 5000:
            return "medium"
        else:
            return "large"
    
    def _get_position(self, bbox: List[int], frame_shape: tuple) -> str:
        """Get object position in frame"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        frame_center_x = frame_shape[1] / 2
        frame_center_y = frame_shape[0] / 2
        
        horizontal = "left" if center_x < frame_center_x - 100 else "right" if center_x > frame_center_x + 100 else "center"
        vertical = "top" if center_y < frame_center_y - 100 else "bottom" if center_y > frame_center_y + 100 else "middle"
        
        return f"{vertical} {horizontal}"