import random
import time
from typing import Dict, List
import cv2
import numpy as np

class VLMProcessor:
    def __init__(self, config):
        self.config = config
        self.setup_vlm()
        
    def setup_vlm(self):
        """Setup VLM (using OpenAI GPT-4V or local alternative)"""
        try:
            # Try to use OpenAI GPT-4 Vision
            import openai
            openai.api_key = self.config.OPENAI_API_KEY
            self.vlm_type = "gpt4v"
            print("✅ GPT-4 Vision model ready")
        except:
            # Fallback to local VLM or descriptive model
            self.vlm_type = "descriptive"
            print("⚠️  Using enhanced descriptive model")
    
    def generate_vlm_description(self, frame: np.ndarray, bbox: List[int], label: str, attributes: Dict) -> str:
        """Generate detailed description using VLM with futuristic style"""
        x1, y1, x2, y2 = bbox
        object_roi = frame[y1:y2, x1:x2]
        
        if self.vlm_type == "gpt4v":
            return self._gpt4v_description(object_roi, label, attributes)
        else:
            return self._enhanced_descriptive_fallback(label, attributes)
    
    def _gpt4v_description(self, roi: np.ndarray, label: str, attributes: Dict) -> str:
        """Use GPT-4 Vision for detailed futuristic description"""
        try:
            import openai
            import base64
            
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', roi)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe this {label} in a concise, futuristic, technical style like a sci-fi AI system. Focus on visual characteristics, appearance, and notable features. Use technical terms and keep it under 2 sentences."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            description = response.choices[0].message.content
            return self._add_futuristic_formatting(description, label, attributes)
            
        except Exception as e:
            print(f"GPT-4V error: {e}")
            return self._enhanced_descriptive_fallback(label, attributes)
    
    def _enhanced_descriptive_fallback(self, label: str, attributes: Dict) -> str:
        """Generate enhanced descriptive fallback with futuristic style"""
        color = attributes.get('color', 'unknown')
        size = attributes.get('size', 'unknown')
        position = attributes.get('position', 'unknown')
        
        # Enhanced futuristic descriptive templates
        descriptions = {
            'person': [
                f"BIOLOGICAL ENTITY: Human subject with {color} attire. Position: {position}. Build: {size}.",
                f"ORGANIC LIFE FORM: Homo sapiens in {color} clothing. Location: {position}. Physical metrics: {size}.",
                f"SENTIENT BEING: Human presence confirmed. Attire: {color}. Spatial coordinates: {position}.",
                f"CARBON-BASED LIFE: Human entity. Visual spectrum: {color}. Area: {position}."
            ],
            'car': [
                f"VEHICLE UNIT: {color} automobile. Dimensions: {size}. Position: {position}.",
                f"TRANSPORTATION DEVICE: {color} automotive unit. Size classification: {size}. Location: {position}.",
                f"MOBILE PLATFORM: {color} vehicle. Scale: {size}. Spatial data: {position}.",
                f"AUTOMOTIVE ENTITY: {color} transport. Magnitude: {size}. Coordinates: {position}."
            ],
            'laptop': [
                f"COMPUTING DEVICE: {color} portable computational unit. Size: {size}. Position: {position}.",
                f"ELECTRONIC APPARATUS: {color} mobile computing device. Scale: {size}. Location: {position}.",
                f"DIGITAL INTERFACE: {color} personal computer system. Dimensions: {size}. Area: {position}.",
                f"TECHNOLOGY UNIT: {color} portable computer. Magnitude: {size}. Spatial data: {position}."
            ],
            'chair': [
                f"SEATING APPARATUS: {color} ergonomic support structure. Size: {size}. Position: {position}.",
                f"FURNITURE UNIT: {color} seating device. Dimensions: {size}. Location: {position}.",
                f"SUPPORT STRUCTURE: {color} chair. Scale: {size}. Coordinates: {position}.",
                f"COMFORT DEVICE: {color} seating furniture. Magnitude: {size}. Area: {position}."
            ],
            'cell phone': [
                f"COMMUNICATION DEVICE: {color} mobile telephony unit. Size: {size}. Position: {position}.",
                f"WIRELESS INTERFACE: {color} smartphone. Dimensions: {size}. Location: {position}.",
                f"PORTABLE TERMINAL: {color} mobile communication device. Scale: {size}. Coordinates: {position}.",
                f"DIGITAL COMMUNICATOR: {color} cellular device. Magnitude: {size}. Spatial data: {position}."
            ]
        }
        
        # Find the best matching description
        for key in descriptions:
            if key in label.lower():
                description = random.choice(descriptions[key])
                return self._add_futuristic_formatting(description, label, attributes)
        
        # Generic futuristic description
        generic_descriptions = [
            f"OBJECT DETECTED: {color} {label}. Size: {size}. Position: {position}.",
            f"ENTITY IDENTIFIED: {label} with {color} signature. Scale: {size}. Location: {position}.",
            f"TARGET ACQUIRED: {color} {label}. Dimensions: {size}. Coordinates: {position}.",
            f"SCAN COMPLETE: {label}. Visual data: {color}. Magnitude: {size}. Area: {position}."
        ]
        
        description = random.choice(generic_descriptions)
        return self._add_futuristic_formatting(description, label, attributes)
    
    def _add_futuristic_formatting(self, description: str, label: str, attributes: Dict) -> str:
        """Add futuristic formatting and metadata to description"""
        # Add confidence if available
        confidence = attributes.get('confidence', 0.0)
        
        # Add technical metadata
        metadata = []
        if attributes.get('color') != 'unknown':
            metadata.append(f"COLOR: {attributes['color'].upper()}")
        if attributes.get('size') != 'unknown':
            metadata.append(f"SIZE: {attributes['size'].upper()}")
        if attributes.get('position') != 'unknown':
            metadata.append(f"POSITION: {attributes['position'].upper()}")
        
        # Add category classification
        category = self._classify_object(label)
        metadata.append(f"CATEGORY: {category}")
        
        # Add confidence level
        if confidence > 0.8:
            confidence_level = "HIGH CONFIDENCE"
        elif confidence > 0.5:
            confidence_level = "MODERATE CONFIDENCE"
        else:
            confidence_level = "LOW CONFIDENCE"
        metadata.append(f"STATUS: {confidence_level}")
        
        # Format final description
        formatted_description = f"{description}"
        if metadata:
            formatted_description += f" | {' | '.join(metadata)}"
        
        return formatted_description
    
    def _classify_object(self, label: str) -> str:
        """Classify objects into futuristic categories"""
        label_lower = label.lower()
        
        if any(obj in label_lower for obj in ['person', 'man', 'woman', 'child']):
            return "BIOLOGICAL ENTITY"
        elif any(obj in label_lower for obj in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            return "TRANSPORTATION UNIT"
        elif any(obj in label_lower for obj in ['laptop', 'cell phone', 'keyboard', 'mouse', 'tv', 'monitor']):
            return "ELECTRONIC DEVICE"
        elif any(obj in label_lower for obj in ['book', 'backpack', 'chair', 'table', 'desk']):
            return "ACADEMIC TOOL"
        elif any(obj in label_lower for obj in ['bottle', 'cup', 'bowl', 'spoon', 'fork']):
            return "CONSUMPTION ITEM"
        elif any(obj in label_lower for obj in ['plant', 'tree', 'flower']):
            return "BOTANICAL ELEMENT"
        else:
            return "GENERAL OBJECT"

class DescriptionGenerator:
    def __init__(self, config):
        self.config = config
        self.vlm_processor = VLMProcessor(config)
        self.last_description_time = 0
        
        # Legacy descriptions for compatibility
        self.legacy_descriptions = {
            'person': [
                "BIOLOGICAL ENTITY: Human subject detected. Neural activity patterns normal.",
                "ORGANIC LIFE FORM: Homo sapiens. Behavioral analysis engaged.",
                "SENTIENT BEING: Human presence confirmed. Threat assessment: minimal.",
                "CARBON-BASED LIFE: Human entity. Social interaction protocols active."
            ],
            'laptop': [
                "COMPUTING DEVICE: Portable computational unit. Processing capabilities: high.",
                "ELECTRONIC APPARATUS: Mobile computing device. Network connectivity active.",
                "DIGITAL INTERFACE: Personal computer system. Data processing engaged.",
                "TECHNOLOGY UNIT: Portable computer. Processing power: adequate for academic tasks."
            ],
            'cell phone': [
                "COMMUNICATION DEVICE: Mobile telephony unit. Signal strength: optimal.",
                "WIRELESS INTERFACE: Smartphone detected. Multiple frequency bands active.",
                "PORTABLE TERMINAL: Mobile communication device. Network: connected.",
                "DIGITAL COMMUNICATOR: Cellular device. Data transmission: stable."
            ],
            'book': [
                "INFORMATION STORAGE: Printed knowledge repository. Content analysis available.",
                "EDUCATIONAL MATERIAL: Bound literary work. Academic relevance: high.",
                "TEXTUAL RESOURCE: Printed pages containing structured information.",
                "LEARNING TOOL: Physical book. Knowledge transfer efficiency: optimal."
            ],
            'chair': [
                "SEATING APPARATUS: Ergonomic support structure. Occupancy status: available.",
                "FURNITURE UNIT: Seating device designed for human comfort.",
                "SUPPORT STRUCTURE: Chair detected. Weight capacity: standard.",
                "COMFORT DEVICE: Seating furniture. Posture support: adequate."
            ]
        }
        
    def generate_description(self, frame: np.ndarray, detection: Dict) -> str:
        """Generate enhanced description using VLM for focused object"""
        try:
            # Extract attributes from detection
            attributes = detection.get('attributes', {})
            attributes['confidence'] = detection['confidence']
            
            # Generate VLM-powered description
            description = self.vlm_processor.generate_vlm_description(
                frame, 
                detection['bbox'], 
                detection['label'], 
                attributes
            )
            
            self.last_description_time = time.time()
            return description
            
        except Exception as e:
            print(f"VLM description generation error: {e}")
            # Fallback to legacy description
            return self._generate_legacy_description(detection['label'], detection['confidence'])
    
    def _generate_legacy_description(self, object_label: str, confidence: float) -> str:
        """Fallback to legacy description system"""
        object_lower = object_label.lower()
        
        # Try exact match first
        for key, descriptions in self.legacy_descriptions.items():
            if key in object_lower:
                description = random.choice(descriptions)
                return f"{description} [CONF: {confidence:.1%}]"
        
        # Fallback template
        fallback_templates = [
            "TARGET ACQUIRED: {object}. Confidence level: {confidence:.1%}",
            "OBJECT IDENTIFIED: {object}. Recognition certainty: {confidence:.1%}",
            "SCAN COMPLETE: {object}. Verification rating: {confidence:.1%}",
            "ANALYSIS: {object}. Detection reliability: {confidence:.1%}"
        ]
        
        template = random.choice(fallback_templates)
        description = template.format(object=object_label.upper(), confidence=confidence)
        
        # Add category classification
        category = self.vlm_processor._classify_object(object_label)
        description += f" | CATEGORY: {category}"
        
        # Add confidence-based qualifiers
        if confidence > 0.8:
            description += " | STATUS: HIGH CONFIDENCE"
        elif confidence > 0.5:
            description += " | STATUS: MODERATE CONFIDENCE"
        else:
            description += " | STATUS: LOW CONFIDENCE"
            
        return description
    
    def generate_quick_description(self, detection: Dict) -> str:
        """Generate quick description without VLM processing"""
        return self._generate_legacy_description(detection['label'], detection['confidence'])