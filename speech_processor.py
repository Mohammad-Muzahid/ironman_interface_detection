import speech_recognition as sr
import pyttsx3
import threading
import time
import numpy as np
from typing import Dict, List, Optional
import re

class AdvancedSpeechProcessor:
    def __init__(self, config):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = pyttsx3.init()
        self.has_microphone = False
        self.is_listening = False
        self.is_listening_for_wake_word = True
        self.last_confidence = 0.8  # Default confidence for visualizer
        
        # Speech recognition settings
        self.energy_threshold = getattr(config, 'ENERGY_THRESHOLD', 1000)
        self.pause_threshold = getattr(config, 'PAUSE_THRESHOLD', 0.8)
        self.speech_timeout = getattr(config, 'SPEECH_TIMEOUT', 8)
        self.wake_word = getattr(config, 'WAKE_WORD', 'orion').lower()
        
        # Initialize microphone
        self._setup_microphone()
        
        # Configure TTS
        self._setup_tts()
        
        # Question patterns for natural language understanding
        self.question_patterns = {
            'what': ['what is', 'what are', 'what does', 'what do', 'what\'s'],
            'where': ['where is', 'where are', 'where\'s'],
            'who': ['who is', 'who are', 'who\'s'],
            'how': ['how many', 'how much', 'how is', 'how are'],
            'why': ['why is', 'why are', 'why does'],
            'when': ['when is', 'when are', 'when does'],
            'which': ['which is', 'which are'],
            'describe': ['describe', 'tell me about', 'explain'],
            'count': ['how many', 'count the', 'number of'],
            'identify': ['what is this', 'what are these', 'identify'],
            'activity': ['what is doing', 'what are they doing', 'what is happening', 'what activity'],
            'relationship': ['carrying', 'holding', 'has', 'with', 'using', 'interacting'],
            'location': ['where is', 'where are', 'location of', 'position of'],
            'color': ['what color', 'what colour', 'color of', 'colour of'],
            'presence': ['is there', 'are there', 'do you see', 'can you see']
        }
        
        print("🎤 Advanced Speech Processor Initialized")
    
    def _setup_microphone(self):
        """Setup microphone with error handling"""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.has_microphone = True
            print("✅ Microphone configured successfully")
            
        except Exception as e:
            print(f"❌ Microphone setup failed: {e}")
            self.has_microphone = False
    
    def _setup_tts(self):
        """Setup text-to-speech engine"""
        try:
            voices = self.tts_engine.getProperty('voices')
            if len(voices) > 0:
                self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 180)  # Speech rate
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            print("✅ TTS engine configured")
            
        except Exception as e:
            print(f"❌ TTS setup failed: {e}")
    
    def listen_for_wake_word(self) -> bool:
        """Listen for the wake word with visual feedback support"""
        if not self.has_microphone:
            return False
        
        try:
            print(f"🎤 Listening for wake word: '{self.wake_word}'...")
            self.is_listening_for_wake_word = True
            
            with self.microphone as source:
                # Listen for wake word with shorter timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
            # Recognize speech
            text = self.recognizer.recognize_google(audio).lower()
            print(f"🔍 Heard: '{text}'")
            
            # Check for wake word
            if self.wake_word in text:
                print(f"✅ Wake word '{self.wake_word}' detected!")
                self.last_confidence = 0.9  # High confidence for wake word
                return True
            else:
                self.last_confidence = 0.3  # Low confidence for non-wake word
                return False
                
        except sr.WaitTimeoutError:
            # No speech detected, continue listening
            return False
        except sr.UnknownValueError:
            # Speech not understood
            self.last_confidence = 0.1
            return False
        except Exception as e:
            print(f"❌ Wake word listening error: {e}")
            return False
    
    def listen_for_command(self) -> Optional[str]:
        """Listen for voice command after wake word with confidence tracking"""
        if not self.has_microphone:
            return None
        
        try:
            print("🎤 Speak your question now...")
            self.is_listening = True
            
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.speech_timeout,
                    phrase_time_limit=10
                )
            
            # Recognize speech with confidence (if supported)
            try:
                # Try to get confidence from Google Speech Recognition
                result = self.recognizer.recognize_google(audio, show_all=True)
                
                if isinstance(result, dict) and 'alternative' in result:
                    # Get the best result with confidence
                    best_match = result['alternative'][0]
                    text = best_match.get('transcript', '')
                    confidence = best_match.get('confidence', 0.5)
                    self.last_confidence = confidence
                else:
                    # Fallback to simple recognition
                    text = self.recognizer.recognize_google(audio)
                    self.last_confidence = 0.7  # Default confidence
                    
            except:
                # Fallback if confidence not available
                text = self.recognizer.recognize_google(audio)
                self.last_confidence = 0.7  # Default confidence
            
            print(f"✅ Question received: '{text}'")
            print(f"📊 Speech confidence: {self.last_confidence:.2f}")
            
            return text
            
        except sr.WaitTimeoutError:
            print("❌ Listening timeout")
            self.last_confidence = 0.0
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            self.last_confidence = 0.0
            return None
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            self.last_confidence = 0.0
            return None
        finally:
            self.is_listening = False
    
    def trigger_listening(self) -> Optional[str]:
        """Manual trigger for listening (spacebar press)"""
        print("🎤 Manual trigger activated...")
        return self.listen_for_command()
    
    def speak(self, text: str):
        """Speak text with error handling"""
        try:
            # Clean and prepare text for speech
            clean_text = self._clean_speech_text(text)
            
            print(f"🗣️ Speaking: {clean_text}")
            
            # Use threading to avoid blocking
            def speak_thread():
                try:
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"❌ TTS error: {e}")
            
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def _clean_speech_text(self, text: str) -> str:
        """Clean and optimize text for speech synthesis"""
        # Remove markdown and special characters
        clean_text = re.sub(r'[\[\]\(\){}]', '', text)
        
        # Replace common abbreviations
        replacements = {
            'vlm': 'V L M',
            'ai': 'A I',
            'cpu': 'C P U',
            'gpu': 'G P U',
            'fps': 'F P S',
            'rgb': 'R G B',
            'hsv': 'H S V',
            'yolo': 'Y O L O',
            'blip': 'B L I P',
            'llava': 'L L A V A'
        }
        
        for abbr, replacement in replacements.items():
            clean_text = re.sub(rf'\b{abbr}\b', replacement, clean_text, flags=re.IGNORECASE)
        
        # Limit length for natural speech
        if len(clean_text) > 200:
            sentences = clean_text.split('.')
            if len(sentences) > 0:
                clean_text = sentences[0] + '.'
                if len(clean_text) < 50 and len(sentences) > 1:
                    clean_text += ' ' + sentences[1] + '.'
        
        return clean_text.strip()
    
    def understand_complex_query(self, question: str, scene_data: Dict) -> Dict:
        """Analyze complex natural language queries"""
        question_lower = question.lower()
        
        analysis = {
            'original_question': question,
            'question_type': 'general',
            'requires_vlm': False,
            'target_objects': [],
            'spatial_queries': [],
            'activity_queries': [],
            'counting_queries': [],
            'color_queries': [],
            'relationship_queries': [],
            'confidence': self.last_confidence
        }
        
        # Detect question type
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    analysis['question_type'] = q_type
                    break
        
        # Extract target objects from scene
        if scene_data and 'objects' in scene_data:
            scene_objects = [obj['label'].lower() for obj in scene_data['objects']]
            
            # Check for object mentions in question
            for obj_label in scene_objects:
                if obj_label in question_lower:
                    analysis['target_objects'].append(obj_label)
            
            # If no specific objects mentioned, use all detected objects
            if not analysis['target_objects'] and scene_objects:
                analysis['target_objects'] = scene_objects[:3]  # Limit to top 3
        
        # Detect spatial queries
        spatial_keywords = ['left', 'right', 'top', 'bottom', 'above', 'below', 
                           'beside', 'next to', 'near', 'far from', 'center', 'middle']
        for keyword in spatial_keywords:
            if keyword in question_lower:
                analysis['spatial_queries'].append(keyword)
        
        # Detect activity queries
        activity_keywords = ['doing', 'sitting', 'standing', 'walking', 'running', 
                           'holding', 'using', 'reading', 'writing', 'watching']
        for keyword in activity_keywords:
            if keyword in question_lower:
                analysis['activity_queries'].append(keyword)
        
        # Detect counting queries
        if 'how many' in question_lower or 'count' in question_lower:
            analysis['counting_queries'].append('quantity')
        
        # Detect color queries
        if 'color' in question_lower or 'colour' in question_lower:
            analysis['color_queries'].append('color')
        
        # Detect relationship queries
        relationship_keywords = ['with', 'and', 'together', 'relationship', 'interaction']
        for keyword in relationship_keywords:
            if keyword in question_lower:
                analysis['relationship_queries'].append(keyword)
        
        # Determine if VLM is required
        complex_queries = (analysis['spatial_queries'] or 
                          analysis['activity_queries'] or 
                          analysis['relationship_queries'] or
                          'describe' in analysis['question_type'] or
                          'explain' in question_lower)
        
        analysis['requires_vlm'] = complex_queries
        
        print(f"🔍 Query Analysis: {analysis}")
        return analysis
    
    def generate_contextual_response(self, query_analysis: Dict, scene_data: Dict, vlm_processor) -> str:
        """Generate contextual response based on query analysis and scene data"""
        question_type = query_analysis['question_type']
        target_objects = query_analysis['target_objects']
        
        # Get basic scene context
        object_count = len(scene_data.get('objects', []))
        people_count = len([obj for obj in scene_data.get('objects', []) 
                           if obj['label'] == 'person'])
        
        # Generate appropriate response based on question type
        if question_type in ['what', 'describe']:
            if target_objects:
                if len(target_objects) == 1:
                    return f"I can see a {target_objects[0]} in the scene."
                else:
                    return f"I can see {', '.join(target_objects[:-1])} and {target_objects[-1]} in the scene."
            else:
                return f"I'm analyzing the scene with {object_count} objects detected."
        
        elif question_type == 'where':
            if target_objects:
                return f"I can locate the {target_objects[0]} using object detection."
            else:
                return "I can help you locate objects in the camera view."
        
        elif question_type == 'how' and 'many' in query_analysis['original_question'].lower():
            if 'person' in query_analysis['original_question'].lower():
                return f"I can detect {people_count} people in the scene."
            else:
                return f"I can count {object_count} objects in total."
        
        elif question_type == 'who':
            return "I can identify people and their activities in the scene."
        
        elif question_type in ['why', 'when', 'which']:
            return "I understand your question. Let me analyze the scene more deeply."
        
        else:
            return "I'm processing your question about the visual scene."
    
    def get_speech_confidence(self) -> float:
        """Get current speech recognition confidence for visualizer"""
        return self.last_confidence
    
    def update_confidence(self, confidence: float):
        """Update speech confidence (for manual updates)"""
        self.last_confidence = max(0.0, min(1.0, confidence))
    
    def stop_listening(self):
        """Stop all listening activities"""
        self.is_listening = False
        self.is_listening_for_wake_word = False
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'tts_engine'):
                self.tts_engine.stop()
        except:
            pass

# Test the speech processor
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    speech_processor = AdvancedSpeechProcessor(config)
    
    print("\n🧪 Testing Speech Processor...")
    
    # Test TTS
    print("1. Testing Text-to-Speech...")
    speech_processor.speak("Hello! I am orion, your advanced visual assistant.")
    
    # Test wake word detection (simulated)
    print("2. Testing wake word analysis...")
    test_questions = [
        "ORION what do you see?",
        "Where is my phone?",
        "How many people are there?",
        "What is the person doing?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        analysis = speech_processor.understand_complex_query(question, {})
        print(f"📊 Analysis: {analysis['question_type']} - Requires VLM: {analysis['requires_vlm']}")
    
    print("\n✅ Speech Processor Test Complete!")