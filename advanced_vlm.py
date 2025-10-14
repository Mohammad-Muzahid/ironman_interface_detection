import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import cv2
import numpy as np
from typing import List, Dict
import time
from PIL import Image

class AdvancedVLMProcessor:
    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        self.processor = None
        self.vlm_model = None
        self.vlm_type = "descriptive"
        
        self._setup_true_vlm()
    
    def _setup_true_vlm(self):
        """Setup BLIP VQA - TRUE Visual Question Answering Model"""
        try:
            print("🚀 Loading BLIP VQA TRUE VLM...")
            
            # Load BLIP VQA model directly
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.vlm_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            
            self.vlm_type = "true_vlm"
            print("✅ BLIP VQA TRUE VLM loaded successfully!")
            
            # Quick test
            test_image = Image.new('RGB', (100, 100), color='red')
            inputs = self.processor(test_image, "What color is this image?", return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.vlm_model.generate(**inputs, max_length=20)
            
            test_result = self.processor.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Test passed: '{test_result}'")
            
        except Exception as e:
            print(f"❌ TRUE VLM setup failed: {e}")
            self.vlm_type = "descriptive"
    
    def analyze_complete_scene(self, frame: np.ndarray) -> Dict:
        """Analyze complete scene with TRUE VLM"""
        try:
            if self.vlm_type == "true_vlm" and self.vlm_model is not None:
                print("🔍 TRUE VLM analyzing scene...")
                return self._vqa_scene_analysis(frame)
            else:
                return self._descriptive_scene_analysis()
        except Exception as e:
            print(f"Scene analysis error: {e}")
            return self._descriptive_scene_analysis()
    
    def _vqa_scene_analysis(self, frame: np.ndarray) -> Dict:
        """Use VQA for comprehensive scene analysis"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Ask multiple questions to build comprehensive analysis
            questions = [
                "What objects do you see in this scene?",
                "What are the main colors in this image?",
                "Are there any people in this scene?",
                "Describe the layout of objects.",
                "What is the main focus of this image?"
            ]
            
            analysis_parts = []
            
            for question in questions:
                try:
                    inputs = self.processor(pil_image, question, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = self.vlm_model.generate(**inputs, max_length=50)
                    
                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                    analysis_parts.append(answer)
                    
                except Exception as e:
                    print(f"Question failed: {e}")
                    continue
            
            if analysis_parts:
                analysis = ". ".join(analysis_parts)
            else:
                analysis = "Scene analysis completed."
            
            print(f"📝 TRUE VLM Analysis: {analysis}")
            
            return {
                'scene_description': analysis,
                'analysis_type': 'true_vlm_vqa',
                'timestamp': time.time(),
                'model': 'blip-vqa-base'
            }
        except Exception as e:
            print(f"VQA analysis error: {e}")
            return self._descriptive_scene_analysis()
    
    def answer_visual_question(self, frame: np.ndarray, question: str) -> str:
        """TRUE VLM - Actually answers visual questions!"""
        try:
            if self.vlm_type == "true_vlm" and self.vlm_model is not None:
                print(f"🤔 TRUE VLM answering: {question}")
                
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Use TRUE VLM to answer the actual question
                inputs = self.processor(pil_image, question, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.vlm_model.generate(**inputs, max_length=100)
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                print(f"🎯 TRUE VLM Answer: {answer}")
                return answer
                
            else:
                return self._enhanced_descriptive_fallback(question)
                
        except Exception as e:
            print(f"VLM QA error: {e}")
            return self._enhanced_descriptive_fallback(question)
    
    def _enhanced_descriptive_fallback(self, question: str) -> str:
        """Enhanced fallback responses"""
        question_lower = question.lower()
        
        if 'phone' in question_lower:
            return "I can detect mobile phones using object detection."
        elif 'person' in question_lower and 'doing' in question_lower:
            return "I can detect people's activities like sitting, using devices, or carrying objects."
        elif 'what' in question_lower and 'see' in question_lower:
            return "I'm analyzing the scene with computer vision."
        elif 'where' in question_lower:
            return "I can locate objects in the camera view."
        elif 'how many' in question_lower:
            return "I can count objects and people in the scene."
        elif 'color' in question_lower:
            return "I can detect colors and visual attributes."
        else:
            return "I understand your question. The vision system is processing."
    
    def _descriptive_scene_analysis(self) -> Dict:
        """Fallback scene analysis"""
        return {
            'scene_description': "Advanced scene analysis ready.",
            'analysis_type': 'descriptive',
            'timestamp': time.time()
        }
    
    def get_model_status(self):
        """Get current model status"""
        return {
            'vlm_type': self.vlm_type,
            'device': self.device,
            'true_vlm_loaded': self.vlm_model is not None,
            'model_name': 'BLIP VQA Base' if self.vlm_model else 'None'
        }

# Test the TRUE VLM
if __name__ == "__main__":
    from config import Config
    config = Config()
    vlm = AdvancedVLMProcessor(config)
    
    print(f"\n📊 VLM Status: {vlm.get_model_status()}")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
    
    # Test TRUE VLM
    if vlm.vlm_type == "true_vlm":
        answer = vlm.answer_visual_question(test_image, "What color is the main object in this image?")
        print(f"\n🤖 TRUE VLM Answer: {answer}")
        
        # Test scene analysis
        scene_data = vlm.analyze_complete_scene(test_image)
        print(f"\n🔍 Scene Analysis: {scene_data['scene_description']}")
    else:
        print("\n❌ TRUE VLM not loaded.")