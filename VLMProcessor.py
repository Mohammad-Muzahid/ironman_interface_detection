import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
import cv2
import numpy as np
from typing import List, Dict, Optional
import time
import openai

class AdvancedVLMProcessor:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_models()
        
    def setup_models(self):
        """Setup advanced VLM models with fallback to GPT-4V"""
        try:
            # Try LLaVA first (better for detailed scene understanding)
            print("🔄 Loading LLaVA model for advanced scene understanding...")
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.vlm_type = "llava"
            print("✅ LLaVA model loaded successfully")
            
        except Exception as e:
            print(f"❌ LLaVA loading failed: {e}")
            try:
                # Fallback to BLIP-2
                print("🔄 Loading BLIP-2 model...")
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.vlm_type = "blip2"
                print("✅ BLIP-2 model loaded successfully")
            except Exception as e2:
                print(f"❌ BLIP-2 loading failed: {e2}")
                # Final fallback to GPT-4V
                try:
                    openai.api_key = self.config.OPENAI_API_KEY
                    self.vlm_type = "gpt4v"
                    print("✅ GPT-4 Vision ready (fallback)")
                except:
                    self.vlm_type = "descriptive"
                    print("⚠️  Using descriptive fallback mode")
    
    def analyze_complete_scene(self, frame: np.ndarray) -> Dict:
        """Analyze complete scene with advanced VLM"""
        try:
            if self.vlm_type in ["llava", "blip2"]:
                return self._vlm_scene_analysis(frame)
            elif self.vlm_type == "gpt4v":
                return self._gpt4v_scene_analysis(frame)
            else:
                return self._descriptive_scene_analysis(frame)
        except Exception as e:
            print(f"Scene analysis error: {e}")
            return self._descriptive_scene_analysis(frame)
    
    def _vlm_scene_analysis(self, frame: np.ndarray) -> Dict:
        """Use open-source VLM for detailed scene analysis"""
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if self.vlm_type == "llava":
            prompt = """Analyze this scene in detail. Describe:
            1. All visible objects and their relationships
            2. Human/animal activities and interactions  
            3. Spatial arrangements and environment
            4. Notable actions or events happening
            Provide a comprehensive but concise analysis."""
        else:  # BLIP-2
            prompt = "Describe this scene in detail including objects, people, activities, and relationships."
        
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.vlm_type == "llava":
                outputs = self.vlm_model.generate(**inputs, max_new_tokens=300)
            else:
                outputs = self.vlm_model.generate(**inputs, max_length=300)
        
        analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'scene_description': analysis,
            'analysis_type': 'vlm_advanced',
            'timestamp': time.time()
        }
    
    def _gpt4v_scene_analysis(self, frame: np.ndarray) -> Dict:
        """Use GPT-4V for scene analysis"""
        try:
            from PIL import Image
            import io
            import base64
            
            # Convert frame to base64
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Analyze this scene comprehensively. Describe all objects, people, activities, relationships, and the overall environment in detail."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return {
                'scene_description': response.choices[0].message.content,
                'analysis_type': 'gpt4v',
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"GPT-4V scene analysis error: {e}")
            return self._descriptive_scene_analysis(frame)
    
    def answer_visual_question(self, frame: np.ndarray, question: str) -> str:
        """Answer questions about visual content"""
        try:
            if self.vlm_type in ["llava", "blip2"]:
                from PIL import Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                inputs = self.processor(images=pil_image, text=question, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    if self.vlm_type == "llava":
                        outputs = self.vlm_model.generate(**inputs, max_new_tokens=150)
                    else:
                        outputs = self.vlm_model.generate(**inputs, max_length=150)
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                return answer
            
            elif self.vlm_type == "gpt4v":
                from PIL import Image
                import io
                import base64
                
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                return response.choices[0].message.content
                
            else:
                return self._descriptive_qa_fallback(question)
                
        except Exception as e:
            print(f"VLM QA error: {e}")
            return f"I cannot analyze that question right now. Error: {str(e)}"
    
    def _descriptive_scene_analysis(self, frame: np.ndarray) -> Dict:
        """Fallback scene analysis"""
        return {
            'scene_description': "Advanced scene analysis unavailable. Using basic object detection.",
            'analysis_type': 'descriptive_fallback',
            'timestamp': time.time()
        }
    
    def _descriptive_qa_fallback(self, question: str) -> str:
        """Fallback for question answering"""
        question_lower = question.lower()
        
        if 'what' in question_lower and 'doing' in question_lower:
            return "Based on visual analysis, I can detect presence but need advanced models for detailed activity recognition."
        elif 'where' in question_lower:
            return "I can detect object locations but need spatial reasoning models for precise positioning."
        elif 'how many' in question_lower:
            return "I can count detected objects but need contextual models for accurate quantification."
        else:
            return "I understand your question but need advanced vision-language capabilities for detailed answers."
    
    # Keep your original methods for backward compatibility
    def generate_vlm_description(self, frame: np.ndarray, bbox: List[int], label: str, attributes: Dict) -> str:
        """Legacy method for single object description"""
        if self.vlm_type in ["llava", "blip2", "gpt4v"]:
            # Use scene analysis for better context
            scene_analysis = self.analyze_complete_scene(frame)
            return f"{label}: {scene_analysis['scene_description'][:100]}..."
        else:
            return self._descriptive_fallback(label, attributes)
    
    def _descriptive_fallback(self, label: str, attributes: Dict) -> str:
        """Your original fallback method"""
        color = attributes.get('color', 'unknown')
        size = attributes.get('size', 'unknown')
        position = attributes.get('position', 'unknown')
        
        descriptions = {
            'person': [
                f"A {size} person wearing {color} clothing positioned at the {position}",
                f"Person in {color} attire, {size} build, located at the {position}",
            ],
            'car': [
                f"A {color} {size} vehicle positioned at the {position}",
                f"{color.capitalize()} car, {size} size, located at the {position}",
            ],
            'chair': [
                f"A {color} {size} chair positioned at the {position}",
                f"{size.capitalize()} {color} chair located at the {position}",
            ]
        }
        
        for key in descriptions:
            if key in label.lower():
                import random
                return random.choice(descriptions[key])
        
        return f"A {color} {size} {label} positioned at the {position}"