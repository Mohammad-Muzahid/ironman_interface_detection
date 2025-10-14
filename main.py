import cv2
import threading
import time
import numpy as np
from typing import List, Dict
from config import Config
from object_detector import AdvancedObjectDetector
from speech_processor import AdvancedSpeechProcessor
from advanced_vlm import AdvancedVLMProcessor
from utils.visualizer import Visualizer
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SIDRAAdvancedVLM:
    def __init__(self, config=Config()):
        self.config = config
        self.detector = AdvancedObjectDetector(config)
        self.speech_processor = AdvancedSpeechProcessor(config)
        self.vlm_processor = AdvancedVLMProcessor(config)
        self.visualizer = Visualizer(config)
        
        # State management
        self.current_frame = None
        self.scene_analysis = {}
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.last_vlm_analysis = 0
        self.vlm_enabled = getattr(config, 'USE_ADVANCED_VLM', True)
        
        # Threading
        self.frame_lock = threading.Lock()
        self.analysis_lock = threading.Lock()
        
        # Store the last question and answer for display
        self.last_question = ""
        self.last_answer = ""
        
        print("🚀 SIDRA Advanced VLM System Initialized")
    
    def start(self):
        """Start the advanced VLM system"""
        print("=" * 70)
        print("🤖 SIDRA ADVANCED VLM - Scene Understanding System")
        print("=" * 70)
        
        print("🎯 Capabilities:")
        print("   • Real-time object detection and tracking")
        print("   • Relationship and activity recognition") 
        print("   • Advanced scene understanding with VLM")
        print("   • Natural language question answering")
        print("   • Dynamic context awareness")
        
        if self.speech_processor.has_microphone:
            print("🎤 Voice: ENABLED - Say 'SIDRA' followed by your question")
            print("💡 Try: 'What is the person doing?' or 'Where is my phone?'")
        else:
            print("⚠️  Voice: DISABLED - Using manual input")
        
        self.is_running = True
        
        # Start processing threads
        threading.Thread(target=self._camera_worker, daemon=True).start()
        threading.Thread(target=self._analysis_worker, daemon=True).start()
        
        if self.speech_processor.has_microphone:
            threading.Thread(target=self._speech_worker, daemon=True).start()
        
        self._display_loop()
    
    def _camera_worker(self):
        """Camera capture thread"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            self.is_running = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.DISPLAY_HEIGHT)
        
        print(f"📷 Camera: {self.config.DISPLAY_WIDTH}x{self.config.DISPLAY_HEIGHT}")
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
                self.frame_count += 1
            else:
                print("❌ Failed to capture frame")
                break
            
            # Update FPS
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_update = current_time
            
            time.sleep(0.01)
        
        cap.release()
        print("📷 Camera stopped")
    
    def _analysis_worker(self):
        """Scene analysis thread"""
        while self.is_running:
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.current_frame.copy()
            
            try:
                # Perform advanced scene analysis
                scene_data = self.detector.detect_with_relationships(frame)
                
                # Periodic VLM analysis (only if enabled)
                current_time = time.time()
                if (self.vlm_enabled and 
                    current_time - self.last_vlm_analysis >= getattr(self.config, 'VLM_PROCESSING_INTERVAL', 2.0)):
                    try:
                        vlm_analysis = self.vlm_processor.analyze_complete_scene(frame)
                        scene_data['vlm_analysis'] = vlm_analysis
                        self.last_vlm_analysis = current_time
                    except Exception as e:
                        print(f"VLM analysis error: {e}")
                
                with self.analysis_lock:
                    self.scene_analysis = scene_data
                    
            except Exception as e:
                print(f"Analysis error: {e}")
            
            # INCREASE THIS SLEEP TIME for better performance:
            time.sleep(0.1)  # Changed from 0.033 to 0.1 (10 FPS instead of 30 FPS)
            
    def _speech_worker(self):
        """Speech processing thread"""
        print("🎤 Speech worker started - Listening for wake word 'SIDRA'...")
        
        while self.is_running:
            try:
                if (self.speech_processor.has_microphone and 
                    self.speech_processor.is_listening_for_wake_word and
                    not self.speech_processor.is_listening):
                    
                    if self.speech_processor.listen_for_wake_word():
                        print("✅ Wake word detected! Listening for question...")
                        question = self.speech_processor.listen_for_command()
                        if question:
                            self._process_visual_question(question)
                            
            except Exception as e:
                print(f"Speech error: {e}")
            time.sleep(0.1)
    
    def _process_visual_question(self, question: str):
        """Process visual questions and store for display"""
        print(f"🔍 Visual Question: '{question}'")
    
         # Store question for display
        self.last_question = question
    
        # Trigger wake word animation
        self.visualizer.trigger_wake_word()
    
        with self.analysis_lock:
             scene_data = self.scene_analysis.copy()
    
        with self.frame_lock:
             current_frame = self.current_frame.copy() if self.current_frame is not None else None
    
        if not scene_data or current_frame is None:
             response = "I don't have visual data to answer that question right now."
        else:
            try:
                # ALWAYS use VLM if available
                if current_frame is not None:
                    response = self.vlm_processor.answer_visual_question(current_frame, question)
                else:
                    response = "No camera feed available."
                
            except Exception as e:
                response = f"Error: {str(e)}"
    
        # Store answer for display
        self.last_answer = response
        print(f"🤖 Answer: {response}")
    
        # Update speech confidence for visualizer
        self.last_speech_confidence = 0.8  # You can get this from speech processor
    
        # Speak the response
        if self.speech_processor.has_microphone:
           self.speech_processor.speak(response)
    
    def _display_loop(self):
        """Main display loop with enhanced visualization"""
        print("\n🎮 Controls:")
        print("   Q = Quit")
        print("   SPACE = Ask question manually") 
        print("   C = Clear display")
        print("   V = Toggle VLM analysis")
        
        while self.is_running:
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                display_frame = self.current_frame.copy()
            
            with self.analysis_lock:
                scene_data = self.scene_analysis.copy()
            
            # Enhanced visualization
            # In the _display_loop method
            if scene_data:
                display_frame = self.visualizer.draw_detections(
                    display_frame, 
                    scene_data.get('objects', []), 
                    None,  # focused_object
                    scene_data  # scene_analysis
        )
            # Show FPS and status in top-left
            status_text = f"FPS: {self.fps} | VLM: {'ON' if self.vlm_enabled else 'OFF'}"
            cv2.putText(display_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show object count if available
            if scene_data.get('scene_context'):
                context = scene_data['scene_context']
                count_text = f"Objects: {context.get('total_objects', 0)} | People: {context.get('total_people', 0)}"
                cv2.putText(display_frame, count_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show last question and answer at the bottom
            if self.last_question:
                # Draw semi-transparent background for text
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, self.config.DISPLAY_HEIGHT - 120), 
                              (self.config.DISPLAY_WIDTH, self.config.DISPLAY_HEIGHT), 
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                # Show question
                question_text = f"Q: {self.last_question}"
                cv2.putText(display_frame, question_text, (10, self.config.DISPLAY_HEIGHT - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Show answer (wrap text if needed)
                if self.last_answer:
                    answer_text = f"A: {self.last_answer}"
                    # Simple text wrapping
                    if len(answer_text) > 80:
                        parts = [answer_text[i:i+80] for i in range(0, len(answer_text), 80)]
                        for i, part in enumerate(parts[:2]):  # Max 2 lines
                            cv2.putText(display_frame, part, (10, self.config.DISPLAY_HEIGHT - 60 + i*25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(display_frame, answer_text, (10, self.config.DISPLAY_HEIGHT - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show instructions at the very bottom
            cv2.putText(display_frame, "SIDRA ADVANCED VLM - Press Q to quit | SPACE for voice", 
                        (10, self.config.DISPLAY_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow(self.config.WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.is_running = False
                print("🛑 Shutting down...")
            elif key == ord(' ') and self.speech_processor.has_microphone:
                print("🎤 Manual trigger activated...")
                question = self.speech_processor.trigger_listening()
                if question:
                    self._process_visual_question(question)
            elif key == ord('c') or key == ord('C'):
                with self.analysis_lock:
                    self.scene_analysis = {}
                # Clear question/answer display
                self.last_question = ""
                self.last_answer = ""
                print("🧹 Display cleared")
            elif key == ord('v') or key == ord('V'):
                self.vlm_enabled = not self.vlm_enabled
                status = "ENABLED" if self.vlm_enabled else "DISABLED"
                print(f"🔮 VLM analysis {status}")
        
        cv2.destroyAllWindows()
        print("👋 SIDRA Advanced VLM shutdown complete")

    def _visualize_advanced_scene(self, frame: np.ndarray, scene_data: Dict) -> np.ndarray:
          """Use the new visualizer for enhanced interface"""
          try:
              # Get detections from scene data
              detections = scene_data.get('objects', [])
        
              # Update visualizer with current data
              if hasattr(self, 'last_speech_confidence'):
                  self.visualizer.update_speech_confidence(self.last_speech_confidence)
        
              # Draw the enhanced interface
              frame = self.visualizer.draw_detections(frame, detections)
        
              # Draw UI overlay (FPS, object count, etc.)
              object_count = len(detections)
              frame = self.visualizer.draw_ui_overlay(frame, object_count, self.fps, None)
        
              # If there's a focused object from VLM, draw focus info
              focused_object = scene_data.get('focused_object')
              if focused_object:
                   description = scene_data.get('focus_description', '')
                   frame = self.visualizer.draw_focus_info(frame, focused_object, description)
        
              return frame

          except Exception as e:
                   print(f"Visualization error: {e}")
                   return frame


    def _get_object_color(self, label: str) -> tuple:
        """Get color for object visualization"""
        label_lower = label.lower()
        
        if 'person' in label_lower:
            return self.config.HOLO_COLORS['human_animal']
        elif any(vehicle in label_lower for vehicle in ['car', 'truck', 'bus']):
            return self.config.HOLO_COLORS['vehicle']
        elif any(elec in label_lower for elec in ['phone', 'laptop', 'tv', 'cell phone']):
            return self.config.HOLO_COLORS['electronic']
        elif any(obj in label_lower for obj in ['chair', 'table', 'book']):
            return self.config.HOLO_COLORS['classroom']
        else:
            return self.config.HOLO_COLORS['default']

def main():
    """Main entry point with enhanced error handling"""
    try:
        print("🤖 Starting SIDRA Advanced VLM System...")
        app = SIDRAAdvancedVLM()
        app.start()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"💥 Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🎯 Thank you for using SIDRA Advanced VLM!")

if __name__ == "__main__":
    main()
