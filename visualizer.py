import cv2
import numpy as np
from typing import List, Dict, Optional
import math
import time

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.animation_time = 0
        self.wake_word_detected = False
        self.wake_word_animation = 0
        self.speech_confidence = 0.0
        self.last_speech_update = 0
        
        # Enhanced Neon Sky Blue color scheme
        self.NEON_SKY_BLUE = self._safe_get_color('sky_blue', (255, 191, 0))
        self.NEON_BRIGHT = (255, 220, 100)
        self.NEON_GLOW = (255, 240, 180)
        self.NEON_DARK = (200, 150, 0)
        self.NEON_CYAN = (255, 255, 100)
        self.NEON_PURPLE = (255, 100, 255)
        self.NEON_GREEN = (100, 255, 100)
        
        # HUD colors
        self.HUD_BLUE = (255, 200, 50)
        self.HUD_CYAN = (255, 255, 150)
        self.HUD_WHITE = (255, 255, 255)
        self.HUD_GLOW = (255, 240, 200)
        
        # Font setup
        self.label_font = cv2.FONT_HERSHEY_SIMPLEX
        self.label_font_scale = getattr(config, 'LABEL_FONT_SCALE', 0.5)
        self.label_thickness = 1
        
        # Detection tracking
        self.previous_positions = {}
        self.position_smoothing = 0.8
        self.human_confidence_history = []
        self.object_confidence_history = []
        
    def _safe_get_color(self, color_name: str, default_color: tuple = (255, 255, 255)) -> tuple:
        """Safely get color from config with fallback"""
        try:
            return self.config.HOLO_COLORS[color_name]
        except (KeyError, AttributeError):
            return default_color
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], focused_object: Dict = None, scene_analysis: Dict = None) -> np.ndarray:
        """Draw enhanced futuristic interface with all new elements"""
        try:
            frame_height, frame_width = frame.shape[:2]
            self.animation_time += 0.03
            
            # Calculate confidence metrics
            self._update_confidence_metrics(detections)
            
            # Draw sci-fi HUD background without gridlines
            frame = self._draw_enhanced_hud_background(frame)
            
            # Draw futuristic HUD frame (like the video reference)
            frame = self._draw_futuristic_hud_frame(frame)
            
            # Draw music equalizer style bars
            frame = self._draw_equalizer_bars(frame)
            
            # Draw bottom concentric circles (moved slightly up)
            frame = self._draw_bottom_concentric_circles(frame)
            
            # Draw VLM analysis description card on right side
            if scene_analysis:
                frame = self._draw_description_card(frame, scene_analysis)
            
            # Process ALL detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Smooth position tracking
                detection_id = f"{detection['label']}_{center_x}_{center_y}"
                if detection_id in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[detection_id]
                    center_x = int(self.position_smoothing * prev_x + (1 - self.position_smoothing) * center_x)
                    center_y = int(self.position_smoothing * prev_y + (1 - self.position_smoothing) * center_y)
                
                self.previous_positions[detection_id] = (center_x, center_y)
                
                # Calculate dynamic scale
                width = x2 - x1
                height = y2 - y1
                object_size = max(width, height)
                scale = self._calculate_dynamic_scale(object_size)
                
                is_focused = focused_object and detection['label'] == focused_object['label']
                
                # Check if it's human/animal (circle) or object (box)
                if any(obj in detection['label'].lower() for obj in self.config.HUMAN_ANIMAL_CLASSES):
                    # Enhanced circle detection with balanced lines
                    frame = self._draw_enhanced_circle_detection(
                        frame, center_x, center_y, x1, y1, x2, y2, scale, is_focused, detection
                    )
                else:
                    # Enhanced box detection
                    frame = self._draw_enhanced_box_detection(
                        frame, x1, y1, x2, y2, scale, is_focused, detection
                    )
            
            # Wake word alert
            if self.wake_word_detected:
                frame = self._draw_wake_word_alert(frame)
                self.wake_word_animation += 1
                if self.wake_word_animation > 60:
                    self.wake_word_detected = False
                    self.wake_word_animation = 0
            
            return frame
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return frame

    def _draw_description_card(self, frame: np.ndarray, scene_analysis: Dict) -> np.ndarray:
        """Draw VLM analysis description card on the right side"""
        frame_height, frame_width = frame.shape[:2]
        
        # Card dimensions - exactly 20mm x 30mm equivalent
        # For 1280x720 resolution, using 3.8 pixels per mm
        card_width = 76   # 20mm * 3.8 = 76 pixels
        card_height = 114 # 30mm * 3.8 = 114 pixels
        
        # Position card on right side, vertically centered within the HUD frame
        margin = 30
        card_x = frame_width - card_width - margin
        card_y = (frame_height - card_height) // 2  # Perfect vertical center
        
        # Get analysis text from VLM
        analysis_text = self._get_analysis_text(scene_analysis)
        
        # Create the card with neon sky blue low opacity
        card_overlay = frame.copy()
        
        # Draw main card background with low opacity
        cv2.rectangle(card_overlay, 
                     (card_x, card_y), 
                     (card_x + card_width, card_y + card_height), 
                     self.NEON_SKY_BLUE, -1, cv2.LINE_AA)
        
        # Draw glowing border
        cv2.rectangle(card_overlay, 
                     (card_x, card_y), 
                     (card_x + card_width, card_y + card_height), 
                     self.NEON_GLOW, 2, cv2.LINE_AA)
        
        # Add futuristic corner brackets
        corner_size = 8
        # Top-left corner
        cv2.line(card_overlay, (card_x, card_y), (card_x + corner_size, card_y), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        cv2.line(card_overlay, (card_x, card_y), (card_x, card_y + corner_size), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        # Top-right corner
        cv2.line(card_overlay, (card_x + card_width, card_y), 
                 (card_x + card_width - corner_size, card_y), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        cv2.line(card_overlay, (card_x + card_width, card_y), 
                 (card_x + card_width, card_y + corner_size), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(card_overlay, (card_x, card_y + card_height), 
                 (card_x + corner_size, card_y + card_height), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        cv2.line(card_overlay, (card_x, card_y + card_height), 
                 (card_x, card_y + card_height - corner_size), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(card_overlay, (card_x + card_width, card_y + card_height), 
                 (card_x + card_width - corner_size, card_y + card_height), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        cv2.line(card_overlay, (card_x + card_width, card_y + card_height), 
                 (card_x + card_width, card_y + card_height - corner_size), 
                 self.NEON_GLOW, 2, cv2.LINE_AA)
        
        # Blend card with frame (low opacity effect)
        frame = cv2.addWeighted(frame, 0.7, card_overlay, 0.3, 0)
        
        # Draw card title
        title = "VLM ANALYSIS"
        title_font_scale = 0.4
        title_thickness = 1
        (title_width, title_height), _ = cv2.getTextSize(title, self.label_font, title_font_scale, title_thickness)
        title_x = card_x + (card_width - title_width) // 2
        title_y = card_y + title_height + 10
        
        cv2.putText(frame, title, (title_x, title_y),
                   self.label_font, title_font_scale, self.NEON_GLOW, title_thickness, cv2.LINE_AA)
        
        # Draw separator line below title
        separator_y = title_y + 5
        cv2.line(frame, 
                 (card_x + 10, separator_y), 
                 (card_x + card_width - 10, separator_y), 
                 self.NEON_SKY_BLUE, 1, cv2.LINE_AA)
        
        # Draw the analysis text with wrapping
        text_start_y = separator_y + 15
        frame = self._draw_wrapped_text(frame, analysis_text, card_x, text_start_y, card_width - 20, card_height - 40)
        
        return frame

    def _get_analysis_text(self, scene_analysis: Dict) -> str:
        """Extract and format analysis text from scene analysis"""
        if not scene_analysis:
            return "No analysis data available"
        
        # Get the main scene description from VLM analysis
        description = scene_analysis.get('scene_description', '')
        analysis_type = scene_analysis.get('analysis_type', 'unknown')
        
        # Clean and format the text for display
        if description:
            # Clean up any markdown or special characters
            clean_text = description.replace('\n', ' ').replace('  ', ' ').strip()
            # Limit length for the card
            if len(clean_text) > 150:
                clean_text = clean_text[:147] + "..."
            return clean_text
        else:
            return f"Analysis type: {analysis_type}\nProcessing visual scene..."

    def _draw_wrapped_text(self, frame: np.ndarray, text: str, start_x: int, start_y: int, 
                          max_width: int, max_height: int) -> np.ndarray:
        """Draw text with proper wrapping within card dimensions"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        # Split text into lines that fit within max_width
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_width = cv2.getTextSize(test_line, self.label_font, 0.3, 1)[0][0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw lines with limited height
        line_height = 12
        current_y = start_y
        font_scale = 0.3
        font_thickness = 1
        
        for i, line in enumerate(lines):
            # Stop if we exceed available height
            if current_y + line_height > start_y + max_height:
                if i > 0:
                    # Add ellipsis to last visible line
                    last_line = lines[i-1]
                    if len(last_line) > 3:
                        last_line = last_line[:-3] + "..."
                    # Redraw the last line with ellipsis
                    (line_width, line_height), _ = cv2.getTextSize(last_line, self.label_font, font_scale, font_thickness)
                    text_x = start_x + (max_width - line_width) // 2
                    cv2.putText(frame, last_line, (text_x, current_y - line_height - 2),
                               self.label_font, font_scale, self.HUD_WHITE, font_thickness, cv2.LINE_AA)
                break
                
            (line_width, line_height), _ = cv2.getTextSize(line, self.label_font, font_scale, font_thickness)
            text_x = start_x + (max_width - line_width) // 2
            
            cv2.putText(frame, line, (text_x, current_y),
                       self.label_font, font_scale, self.HUD_WHITE, font_thickness, cv2.LINE_AA)
            
            current_y += line_height + 2
        
        return frame
    
    def _update_confidence_metrics(self, detections: List[Dict]):
        """Update confidence metrics for performance bars"""
        human_confidences = [det['confidence'] for det in detections 
                           if any(obj in det['label'].lower() for obj in self.config.HUMAN_ANIMAL_CLASSES)]
        object_confidences = [det['confidence'] for det in detections 
                             if not any(obj in det['label'].lower() for obj in self.config.HUMAN_ANIMAL_CLASSES)]
        
        # Update human confidence (average of all human detections)
        if human_confidences:
            avg_human_confidence = sum(human_confidences) / len(human_confidences)
            self.human_confidence_history.append(avg_human_confidence)
            if len(self.human_confidence_history) > 10:
                self.human_confidence_history.pop(0)
        
        # Update object confidence (average of all object detections)
        if object_confidences:
            avg_object_confidence = sum(object_confidences) / len(object_confidences)
            self.object_confidence_history.append(avg_object_confidence)
            if len(self.object_confidence_history) > 10:
                self.object_confidence_history.pop(0)
        
        # Decay speech confidence over time
        current_time = time.time()
        if current_time - self.last_speech_update > 2.0:  # Decay after 2 seconds
            self.speech_confidence *= 0.95
    
    def _draw_enhanced_hud_background(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced HUD without gridlines"""
        frame_height, frame_width = frame.shape[:2]
        hud_frame = frame.copy()
        
        # Create subtle blue overlay (no gridlines)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        overlay[:] = self.HUD_BLUE
        hud_frame = cv2.addWeighted(hud_frame, 0.9, overlay, 0.1, 0)
        
        return hud_frame
    
    def _draw_futuristic_hud_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw futuristic HUD frame with refined positioning"""
        frame_height, frame_width = frame.shape[:2]
        
        # REFINED: Increased top margin to prevent title overlap
        top_margin = 60  # Increased from 30 to provide space for title
        side_margin = 30
        bottom_margin = 30
        
        frame_x1, frame_y1 = side_margin, top_margin
        frame_x2, frame_y2 = frame_width - side_margin, frame_height - bottom_margin
        
        # Draw main frame with double border
        border_thickness = 2
        for i in range(2):
            offset = i * 4
            color = self.NEON_SKY_BLUE if i == 0 else self.NEON_CYAN
            cv2.rectangle(frame, 
                         (frame_x1 - offset, frame_y1 - offset), 
                         (frame_x2 + offset, frame_y2 + offset), 
                         color, border_thickness, cv2.LINE_AA)
        
        # Draw corner brackets (like in the video)
        bracket_size = 25
        bracket_thickness = 3
        
        # Top-left bracket
        cv2.line(frame, (frame_x1, frame_y1), (frame_x1 + bracket_size, frame_y1), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        cv2.line(frame, (frame_x1, frame_y1), (frame_x1, frame_y1 + bracket_size), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        
        # Top-right bracket
        cv2.line(frame, (frame_x2, frame_y1), (frame_x2 - bracket_size, frame_y1), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        cv2.line(frame, (frame_x2, frame_y1), (frame_x2, frame_y1 + bracket_size), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        
        # Bottom-left bracket
        cv2.line(frame, (frame_x1, frame_y2), (frame_x1 + bracket_size, frame_y2), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        cv2.line(frame, (frame_x1, frame_y2), (frame_x1, frame_y2 - bracket_size), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        
        # Bottom-right bracket
        cv2.line(frame, (frame_x2, frame_y2), (frame_x2 - bracket_size, frame_y2), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        cv2.line(frame, (frame_x2, frame_y2), (frame_x2, frame_y2 - bracket_size), 
                self.NEON_GLOW, bracket_thickness, cv2.LINE_AA)
        
        # Draw diagonal lines from center
        center_x = frame_width // 2
        
        # Top frame: center to left diagonal
        cv2.line(frame, (center_x, frame_y1), (frame_x1, frame_y1), 
                self.NEON_SKY_BLUE, 2, cv2.LINE_AA)
        
        # Bottom frame: center to right diagonal  
        cv2.line(frame, (center_x, frame_y2), (frame_x2, frame_y2),
                self.NEON_CYAN, 2, cv2.LINE_AA)
        
        # Draw diagonal corner connectors (plexus style)
        connector_length = 15
        corners = [
            (frame_x1, frame_y1), (frame_x2, frame_y1),
            (frame_x1, frame_y2), (frame_x2, frame_y2)
        ]
        
        for cx, cy in corners:
            for angle in [45, 135, 225, 315]:
                if (cx == frame_x1 and angle in [135, 225]) or (cx == frame_x2 and angle in [45, 315]):
                    continue
                if (cy == frame_y1 and angle in [225, 315]) or (cy == frame_y2 and angle in [45, 135]):
                    continue
                    
                end_x = int(cx + connector_length * math.cos(math.radians(angle)))
                end_y = int(cy + connector_length * math.sin(math.radians(angle)))
                cv2.line(frame, (cx, cy), (end_x, end_y), 
                        self.NEON_SKY_BLUE, 1, cv2.LINE_AA)
        
        # Draw side panel elements with digital patterns
        self._draw_digital_side_panels(frame, frame_x1, frame_y1, frame_x2, frame_y2)
        
        return frame
    
    def _draw_digital_side_panels(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Draw digital side panels with refined positioning inside frame"""
        frame_height, frame_width = frame.shape[:2]
        panel_width = 10  # Slightly narrower for better fit
        
        # REFINED: Position side bars inside the main frame with vertical centering
        left_panel_x = x1 + 15  # Moved inside from frame edge
        right_panel_x = x2 - 25  # Moved inside from frame edge
        
        # Calculate vertical center for side panels
        panel_height = y2 - y1
        panel_center_y = y1 + panel_height // 2
        
        # Left side panel - vertically centered
        for i in range(0, panel_height, 4):
            y_pos = y1 + i
            if y_pos > y2:
                continue
            alpha = 0.3 + 0.4 * math.sin(self.animation_time + i * 0.1)
            color = tuple(int(c * alpha) for c in self.NEON_CYAN)
            cv2.line(frame, 
                    (left_panel_x, y_pos), 
                    (left_panel_x + panel_width, y_pos), 
                    color, 1, cv2.LINE_AA)
        
        # Right side panel - vertically centered
        for i in range(0, panel_height, 4):
            y_pos = y1 + i
            if y_pos > y2:
                continue
            alpha = 0.3 + 0.4 * math.sin(self.animation_time + i * 0.1 + math.pi)
            color = tuple(int(c * alpha) for c in self.NEON_SKY_BLUE)
            cv2.line(frame, 
                    (right_panel_x, y_pos), 
                    (right_panel_x + panel_width, y_pos), 
                    color, 1, cv2.LINE_AA)
        
        # Panel borders - refined positioning
        cv2.rectangle(frame, 
                     (left_panel_x, y1 + 20), 
                     (left_panel_x + panel_width, y2 - 20), 
                     self.NEON_CYAN, 1, cv2.LINE_AA)
        cv2.rectangle(frame, 
                     (right_panel_x, y1 + 20), 
                     (right_panel_x + panel_width, y2 - 20), 
                     self.NEON_SKY_BLUE, 1, cv2.LINE_AA)
    
    def _draw_equalizer_bars(self, frame: np.ndarray) -> np.ndarray:
        """Draw music equalizer style bars with only HUMAN confidence on left side"""
        frame_height, frame_width = frame.shape[:2]
    
        # Calculate current HUMAN confidence value (smoothed)
        human_confidence = np.mean(self.human_confidence_history) if self.human_confidence_history else 0
    
        # Position equalizer bar on LEFT side only (centered vertically)
        left_x = 50  # Left side position
    
        # Calculate vertical center position
        frame_top = 60  # Top margin
        frame_bottom = frame_height - 30  # Bottom margin
        available_height = frame_bottom - frame_top
        base_y = frame_top + (available_height // 2)  # Vertically centered
    
        # Draw only HUMAN confidence equalizer on left side
        self._draw_single_equalizer(frame, left_x, base_y, human_confidence, 
                              self.NEON_SKY_BLUE, "HUMAN")
    
        return frame
    
    def _draw_single_equalizer(self, frame: np.ndarray, x: int, base_y: int, 
                             confidence: float, color: tuple, label: str):
        """Draw a single music equalizer style bar with refined positioning"""
        bar_count = 8
        bar_width = 6
        bar_spacing = 3
        max_bar_height = 80  # Reduced height for better fit
        
        # Calculate bar heights with equalizer effect
        bar_heights = []
        for i in range(bar_count):
            # Create wave pattern based on confidence and bar position
            wave = math.sin(self.animation_time * 3 + i * 0.8)
            height_variation = 0.3 * wave + 0.7
            bar_height = int(max_bar_height * confidence * height_variation)
            bar_heights.append(max(5, bar_height))  # Minimum height
        
        # Draw bars
        for i, height in enumerate(bar_heights):
            bar_x = x + i * (bar_width + bar_spacing)
            bar_y = base_y - height
            
            # Draw bar with gradient
            for j in range(height):
                alpha = (j / height) * 0.8 + 0.2
                bar_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, 
                        (bar_x, base_y - j), 
                        (bar_x + bar_width, base_y - j), 
                        bar_color, 1, cv2.LINE_AA)
            
            # Draw bar cap (glowing top)
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + 3), 
                         self.NEON_GLOW, -1, cv2.LINE_AA)
            
            # Draw bar frame
            cv2.rectangle(frame, 
                         (bar_x, base_y), 
                         (bar_x + bar_width, bar_y), 
                         self.NEON_GLOW, 1, cv2.LINE_AA)
        
        # Draw label
        label_y = base_y + 30
        cv2.putText(frame, label, (x - 10, label_y),
                   self.label_font, 0.4, self.NEON_GLOW, 1, cv2.LINE_AA)
        
        # Draw confidence percentage
        confidence_text = f"{confidence:.0%}"
        text_size = cv2.getTextSize(confidence_text, self.label_font, 0.5, 1)[0]
        text_x = x + (bar_count * (bar_width + bar_spacing) - bar_spacing - text_size[0]) // 2
        cv2.putText(frame, confidence_text, 
                   (text_x, base_y + 50),
                   self.label_font, 0.5, self.NEON_GLOW, 1, cv2.LINE_AA)
    
    def _draw_bottom_concentric_circles(self, frame: np.ndarray) -> np.ndarray:
        """Draw four concentric layered circles at bottom with refined positioning"""
        frame_height, frame_width = frame.shape[:2]
        
        # REFINED: Increased vertical offset to prevent touching frame edge
        vertical_offset = 100  # Increased from 80 for better spacing
        
        # Right-bottom circle (Person confidence)
        right_center_x = frame_width - 120
        right_center_y = frame_height - vertical_offset
        
        # Left-bottom circle (Speech confidence)
        left_center_x = 120
        left_center_y = frame_height - vertical_offset
        
        circle_radius = 45  # Slightly smaller for better spacing
        
        # Draw right concentric circles
        self._draw_concentric_circle_set(frame, right_center_x, right_center_y, circle_radius, 
                                       self.NEON_SKY_BLUE, "PERSON")
        
        # Draw left concentric circles  
        self._draw_concentric_circle_set(frame, left_center_x, left_center_y, circle_radius,
                                       self.NEON_CYAN, "SPEECH")
        
        return frame
    
    def _draw_concentric_circle_set(self, frame: np.ndarray, center_x: int, center_y: int, 
                                  base_radius: int, color: tuple, circle_type: str):
        """Draw a set of four concentric circles with unique designs"""
        
        # Layer 1: Inner solid circle with pulsing effect
        pulse = (math.sin(self.animation_time * 2) + 1) * 0.3 + 0.7
        radius1 = int(base_radius * 0.3 * pulse)
        cv2.circle(frame, (center_x, center_y), radius1, color, -1, cv2.LINE_AA)
        
        # Layer 2: Dashed circle
        radius2 = int(base_radius * 0.5)
        self._draw_dashed_circle(frame, center_x, center_y, radius2, color, 12)
        
        # Layer 3: Rotating segmented circle
        radius3 = int(base_radius * 0.7)
        self._draw_rotating_segments(frame, center_x, center_y, radius3, color, 8)
        
        # Layer 4: Outer ring with orbiting elements
        radius4 = base_radius
        self._draw_orbiting_ring(frame, center_x, center_y, radius4, color, 6)
        
        # Display confidence percentage in center
        if circle_type == "PERSON":
            confidence = np.mean(self.human_confidence_history) if self.human_confidence_history else 0
        else:  # SPEECH
            confidence = self.speech_confidence
            
        confidence_text = f"{confidence:.0%}"
        text_size = cv2.getTextSize(confidence_text, self.label_font, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        # Text background
        bg_padding = 5
        cv2.rectangle(frame, 
                     (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                     (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                     (0, 0, 0), -1, cv2.LINE_AA)
        
        cv2.putText(frame, confidence_text, (text_x, text_y),
                   self.label_font, 0.6, self.NEON_GLOW, 2, cv2.LINE_AA)
        
        # Circle label
        label_y = center_y + base_radius + 20
        cv2.putText(frame, circle_type, 
                   (center_x - 30, label_y),
                   self.label_font, 0.4, self.NEON_GLOW, 1, cv2.LINE_AA)

    def _draw_dashed_circle(self, frame: np.ndarray, center_x: int, center_y: int, 
                          radius: int, color: tuple, num_dashes: int):
        """Draw dashed circle"""
        for i in range(num_dashes):
            if i % 2 == 0:  # Skip every other segment for dashed effect
                continue
            angle1 = i * (2 * math.pi / num_dashes)
            angle2 = (i + 1) * (2 * math.pi / num_dashes)
            start_x = int(center_x + radius * math.cos(angle1))
            start_y = int(center_y + radius * math.sin(angle1))
            end_x = int(center_x + radius * math.cos(angle2))
            end_y = int(center_y + radius * math.sin(angle2))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2, cv2.LINE_AA)
    
    def _draw_rotating_segments(self, frame: np.ndarray, center_x: int, center_y: int, 
                              radius: int, color: tuple, num_segments: int):
        """Draw rotating segmented circle"""
        for i in range(num_segments):
            angle = i * (2 * math.pi / num_segments) + self.animation_time
            start_x = int(center_x + (radius - 3) * math.cos(angle))
            start_y = int(center_y + (radius - 3) * math.sin(angle))
            end_x = int(center_x + (radius + 3) * math.cos(angle))
            end_y = int(center_y + (radius + 3) * math.sin(angle))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2, cv2.LINE_AA)
    
    def _draw_orbiting_ring(self, frame: np.ndarray, center_x: int, center_y: int, 
                          radius: int, color: tuple, num_orbiters: int):
        """Draw ring with orbiting elements"""
        # Main ring
        cv2.circle(frame, (center_x, center_y), radius, color, 2, cv2.LINE_AA)
        
        # Orbiting dots
        for i in range(num_orbiters):
            angle = i * (2 * math.pi / num_orbiters) + self.animation_time * 2
            dot_x = int(center_x + radius * math.cos(angle))
            dot_y = int(center_y + radius * math.sin(angle))
            cv2.circle(frame, (dot_x, dot_y), 3, self.NEON_GLOW, -1, cv2.LINE_AA)

    def _draw_enhanced_circle_detection(self, frame: np.ndarray, center_x: int, center_y: int, 
                                      x1: int, y1: int, x2: int, y2: int, scale: float, 
                                      is_focused: bool, detection: Dict) -> np.ndarray:
        """Enhanced circle detection with balanced, shortened lines"""
        # Draw the 5-layer futuristic circle
        frame = self._draw_multi_layered_circle(frame, center_x, center_y, scale, is_focused, detection)
        
        # Draw balanced labeling line (shorter and more aesthetic)
        frame = self._draw_balanced_circle_label(frame, center_x, center_y, x1, y1, x2, y2, detection, is_focused)
        
        return frame
    
    def _draw_balanced_circle_label(self, frame: np.ndarray, center_x: int, center_y: int,
                                  x1: int, y1: int, x2: int, y2: int, 
                                  detection: Dict, is_focused: bool) -> np.ndarray:
        """Draw balanced, shortened labeling line for circle detections"""
        label = detection['label']
        confidence = detection['confidence']
        
        # Calculate balanced label position (shorter distance)
        bbox_width = x2 - x1
        label_distance = 80 + bbox_width // 3
        
        # Use object-specific angle
        object_hash = hash(label) % 360
        label_angle = math.radians(45 + (object_hash % 90))
        
        # Calculate label position
        label_x = int(center_x + label_distance * math.cos(label_angle))
        label_y = int(center_y + label_distance * math.sin(label_angle))
        
        # Start from outer circle edge (shorter line)
        start_radius = 20 + 45
        start_x = int(center_x + start_radius * math.cos(label_angle))
        start_y = int(center_y + start_radius * math.sin(label_angle))
        
        # Draw balanced connecting line
        self._draw_glowing_line(frame, start_x, start_y, label_x, label_y, self.NEON_SKY_BLUE)
        
        # Draw compact label card
        frame = self._draw_compact_label_card(frame, label_x, label_y, detection, is_focused, "circle")
        
        return frame

    def _draw_enhanced_box_detection(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                                   scale: float, is_focused: bool, detection: Dict) -> np.ndarray:
        """Enhanced box detection with balanced labeling"""
        # Draw futuristic bounding box
        frame = self._draw_futuristic_bbox(frame, detection, is_focused)
        
        # Draw balanced labeling line
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        frame = self._draw_balanced_box_label(frame, center_x, center_y, x1, y1, x2, y2, detection, is_focused)
        
        return frame
    
    def _draw_balanced_box_label(self, frame: np.ndarray, center_x: int, center_y: int,
                               x1: int, y1: int, x2: int, y2: int,
                               detection: Dict, is_focused: bool) -> np.ndarray:
        """Draw balanced labeling line for box detections"""
        label = detection['label']
        confidence = detection['confidence']
        
        # Calculate balanced label position
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        label_distance = 70 + max(bbox_width, bbox_height) // 3
        
        # Use object-specific angle
        object_hash = hash(label) % 360
        label_angle = math.radians(225 + (object_hash % 90))
        
        # Calculate label position
        label_x = int(center_x + label_distance * math.cos(label_angle))
        label_y = int(center_y + label_distance * math.sin(label_angle))
        
        # Start from bbox edge (shorter)
        start_x = int(center_x + (bbox_width // 2 + 5) * math.cos(label_angle))
        start_y = int(center_y + (bbox_height // 2 + 5) * math.sin(label_angle))
        
        # Draw balanced connecting line
        self._draw_glowing_line(frame, start_x, start_y, label_x, label_y, self.NEON_CYAN)
        
        # Draw compact label card
        frame = self._draw_compact_label_card(frame, label_x, label_y, detection, is_focused, "box")
        
        return frame

    def _draw_glowing_line(self, frame: np.ndarray, start_x: int, start_y: int, 
                          end_x: int, end_y: int, color: tuple):
        """Draw a line with glow effect"""
        # Main line (thinner for balance)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1, cv2.LINE_AA)
        
        # Subtle glow effect
        for i in range(1, 2):
            glow_color = tuple(int(c * (0.7 - i * 0.2)) for c in color)
            cv2.line(frame, (start_x, start_y), (end_x, end_y), glow_color, 1, cv2.LINE_AA)
        
        # Smaller arrow head
        arrow_size = 6
        line_angle = math.atan2(end_y - start_y, end_x - start_x)
        
        arrow1_x = int(end_x - arrow_size * math.cos(line_angle - math.pi/6))
        arrow1_y = int(end_y - arrow_size * math.sin(line_angle - math.pi/6))
        arrow2_x = int(end_x - arrow_size * math.cos(line_angle + math.pi/6))
        arrow2_y = int(end_y - arrow_size * math.sin(line_angle + math.pi/6))
        
        cv2.line(frame, (end_x, end_y), (arrow1_x, arrow1_y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (end_x, end_y), (arrow2_x, arrow2_y), color, 1, cv2.LINE_AA)

    def _draw_compact_label_card(self, frame: np.ndarray, label_x: int, label_y: int,
                               detection: Dict, is_focused: bool, detection_type: str) -> np.ndarray:
        """Draw compact label card for balanced appearance"""
        label = detection['label']
        confidence = detection['confidence']
        
        # Simpler text lines for compact design
        lines = [
            f"{label.upper()}",
            f"{confidence:.1%}"
        ]
        
        # Calculate compact dimensions
        line_heights = []
        max_width = 0
        
        for line in lines:
            (width, height), _ = cv2.getTextSize(line, self.label_font, self.label_font_scale, self.label_thickness)
            line_heights.append(height)
            max_width = max(max_width, width)
        
        total_height = sum(line_heights) + 15
        
        # Compact card dimensions
        card_width = max_width + 20
        card_height = total_height + 10
        
        # Adjust position to keep card in frame
        frame_height, frame_width = frame.shape[:2]
        if label_x + card_width > frame_width:
            label_x = frame_width - card_width - 10
        if label_y + card_height > frame_height:
            label_y = frame_height - card_height - 10
        if label_y < 0:
            label_y = 10
        
        # Draw compact neon card
        card_overlay = frame.copy()
        
        # Simple card background
        cv2.rectangle(card_overlay, 
                     (label_x, label_y), 
                     (label_x + card_width, label_y + card_height), 
                     self.NEON_SKY_BLUE, -1, cv2.LINE_AA)
        
        # Card border
        border_color = self._get_category_color(detection['label']) if is_focused else self.NEON_SKY_BLUE
        cv2.rectangle(card_overlay, 
                     (label_x, label_y), 
                     (label_x + card_width, label_y + card_height), 
                     border_color, 1, cv2.LINE_AA)
        
        # Blend card with frame
        frame = cv2.addWeighted(frame, 0.8, card_overlay, 0.2, 0)
        
        # Draw text lines
        current_y = label_y + 20
        for i, line in enumerate(lines):
            color = self.NEON_GLOW if i == 0 else self.HUD_WHITE
            font_scale = self.label_font_scale * (1.1 if i == 0 else 0.8)
            
            (text_width, text_height), _ = cv2.getTextSize(line, self.label_font, font_scale, 1)
            text_x = label_x + (card_width - text_width) // 2
            
            cv2.putText(frame, line, (text_x, current_y),
                       self.label_font, font_scale, color, 1, cv2.LINE_AA)
            
            current_y += text_height + 5
        
        return frame

    def update_speech_confidence(self, confidence: float):
        """Update speech recognition confidence for left-bottom circle"""
        self.speech_confidence = confidence
        self.last_speech_update = time.time()

    def _calculate_dynamic_scale(self, object_size: int) -> float:
        return max(0.6, min(1.8, 150 / object_size))
    
    def _draw_multi_layered_circle(self, frame: np.ndarray, center_x: int, center_y: int, 
                                 scale: float, is_focused: bool, detection: Dict) -> np.ndarray:
        circle_frame = frame.copy()
        base_radius = int(20 * scale)
        
        # Layer 1: Inner Circle
        circle_frame = self._draw_layer1_inner_circle(circle_frame, center_x, center_y, base_radius, is_focused)
        # Layer 2: Box-shaped segments
        circle_frame = self._draw_layer2_box_segments(circle_frame, center_x, center_y, base_radius + 12, is_focused)
        # Layer 3: Thin circular frame
        circle_frame = self._draw_layer3_thin_circle(circle_frame, center_x, center_y, base_radius + 28, is_focused)
        # Layer 4: Half bold frame + half vertical lines
        circle_frame = self._draw_layer4_hybrid(circle_frame, center_x, center_y, base_radius + 40, is_focused)
        # Layer 5: Multiple cut segmented circular layers
        circle_frame = self._draw_layer5_segmented(circle_frame, center_x, center_y, base_radius + 55, is_focused)
        
        alpha = 0.85
        frame = cv2.addWeighted(frame, 1 - alpha, circle_frame, alpha, 0)
        return frame

    def _draw_layer1_inner_circle(self, frame: np.ndarray, center_x: int, center_y: int, 
                                radius: int, is_focused: bool) -> np.ndarray:
        cv2.circle(frame, (center_x, center_y), radius, self.NEON_SKY_BLUE, 3, lineType=cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), radius - 2, self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
        for i in range(4):
            angle = self.animation_time * 2 + (i * math.pi/2)
            dot_x = int(center_x + (radius - 6) * math.cos(angle))
            dot_y = int(center_y + (radius - 6) * math.sin(angle))
            cv2.circle(frame, (dot_x, dot_y), 2, self.NEON_GLOW, -1, lineType=cv2.LINE_AA)
        return frame
    
    def _draw_layer2_box_segments(self, frame: np.ndarray, center_x: int, center_y: int, 
                                radius: int, is_focused: bool) -> np.ndarray:
        num_segments = 16
        box_length = 6
        for i in range(num_segments):
            angle = i * (2 * math.pi / num_segments) + self.animation_time * 0.5
            start_x = int(center_x + (radius - box_length//2) * math.cos(angle))
            start_y = int(center_y + (radius - box_length//2) * math.sin(angle))
            end_x = int(center_x + (radius + box_length//2) * math.cos(angle))
            end_y = int(center_y + (radius + box_length//2) * math.sin(angle))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), self.NEON_SKY_BLUE, 2, lineType=cv2.LINE_AA)
            
            perp_angle = angle + math.pi/2
            perp_length = 3
            
            perp1_x = int(start_x + perp_length * math.cos(perp_angle))
            perp1_y = int(start_y + perp_length * math.sin(perp_angle))
            perp2_x = int(start_x - perp_length * math.cos(perp_angle))
            perp2_y = int(start_y - perp_length * math.sin(perp_angle))
            
            perp3_x = int(end_x + perp_length * math.cos(perp_angle))
            perp3_y = int(end_y + perp_length * math.sin(perp_angle))
            perp4_x = int(end_x - perp_length * math.cos(perp_angle))
            perp4_y = int(end_y - perp_length * math.sin(perp_angle))
            
            cv2.line(frame, (start_x, start_y), (perp1_x, perp1_y), self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (start_x, start_y), (perp2_x, perp2_y), self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (end_x, end_y), (perp3_x, perp3_y), self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (end_x, end_y), (perp4_x, perp4_y), self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
            
            if i % 2 == 0:
                cv2.circle(frame, (start_x, start_y), 1, self.NEON_GLOW, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (end_x, end_y), 1, self.NEON_GLOW, -1, lineType=cv2.LINE_AA)
        return frame

    def _draw_layer3_thin_circle(self, frame: np.ndarray, center_x: int, center_y: int, 
                               radius: int, is_focused: bool) -> np.ndarray:
        cv2.circle(frame, (center_x, center_y), radius, self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
        num_dots = 20
        for i in range(num_dots):
            angle = i * (2 * math.pi / num_dots) + self.animation_time
            pulse = (math.sin(angle * 2 + self.animation_time * 3) + 1) * 0.5
            dot_radius = 1 + int(1 * pulse)
            dot_x = int(center_x + radius * math.cos(angle))
            dot_y = int(center_y + radius * math.sin(angle))
            cv2.circle(frame, (dot_x, dot_y), dot_radius, self.NEON_GLOW, -1, lineType=cv2.LINE_AA)
        return frame
    
    def _draw_layer4_hybrid(self, frame: np.ndarray, center_x: int, center_y: int, 
                          radius: int, is_focused: bool) -> np.ndarray:
        start_angle_bold = math.pi/2
        end_angle_bold = 3*math.pi/2
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 
                   math.degrees(start_angle_bold), math.degrees(end_angle_bold), 
                   self.NEON_SKY_BLUE, 3, lineType=cv2.LINE_AA)
        start_angle_lines = -math.pi/2
        end_angle_lines = math.pi/2
        num_lines = 16
        for i in range(num_lines):
            angle = start_angle_lines + (i * (end_angle_lines - start_angle_lines) / num_lines)
            inner_radius = radius - 4
            outer_radius = radius + 4
            inner_x = int(center_x + inner_radius * math.cos(angle))
            inner_y = int(center_y + inner_radius * math.sin(angle))
            outer_x = int(center_x + outer_radius * math.cos(angle))
            outer_y = int(center_y + outer_radius * math.sin(angle))
            cv2.line(frame, (inner_x, inner_y), (outer_x, outer_y), 
                    self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
        return frame
    
    def _draw_layer5_segmented(self, frame: np.ndarray, center_x: int, center_y: int, 
                             radius: int, is_focused: bool) -> np.ndarray:
        cv2.circle(frame, (center_x, center_y), radius, self.NEON_SKY_BLUE, 2, lineType=cv2.LINE_AA)
        inner_radius = radius - 5
        num_segments = 32
        for i in range(num_segments):
            if i % 4 == 0:
                continue
            angle1 = i * (2 * math.pi / num_segments) + self.animation_time
            angle2 = (i + 1) * (2 * math.pi / num_segments) + self.animation_time
            start_x = int(center_x + inner_radius * math.cos(angle1))
            start_y = int(center_y + inner_radius * math.sin(angle1))
            end_x = int(center_x + inner_radius * math.cos(angle2))
            end_y = int(center_y + inner_radius * math.sin(angle2))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), self.NEON_BRIGHT, 1, lineType=cv2.LINE_AA)
        num_orbiters = 12
        orbiter_radius = radius + 8
        for i in range(num_orbiters):
            angle = i * (2 * math.pi / num_orbiters) + self.animation_time * 1.5
            orbiter_x = int(center_x + orbiter_radius * math.cos(angle))
            orbiter_y = int(center_y + orbiter_radius * math.sin(angle))
            cross_size = 2
            cv2.line(frame, (orbiter_x - cross_size, orbiter_y), (orbiter_x + cross_size, orbiter_y), self.NEON_GLOW, 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (orbiter_x, orbiter_y - cross_size), (orbiter_x, orbiter_y + cross_size), self.NEON_GLOW, 1, lineType=cv2.LINE_AA)
        return frame

    def _draw_futuristic_bbox(self, frame: np.ndarray, detection: Dict, is_focused: bool) -> np.ndarray:
        x1, y1, x2, y2 = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        color = self._get_category_color(label)
        for i in range(3, 0, -1):
            glow_color = [int(c * 0.8) for c in color]
            cv2.rectangle(frame, (x1-i, y1-i), (x2+i, y2+i), glow_color, 1)
        thickness = 3 if is_focused else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        marker_length = 15
        self._draw_corners(frame, x1, y1, x2, y2, marker_length, color)
        return frame

    def _get_category_color(self, label: str) -> tuple:
        label_lower = label.lower()
        try:
            if any(obj in label_lower for obj in self.config.HUMAN_ANIMAL_CLASSES):
                return self._safe_get_color('human_animal', (255, 255, 0))
            elif any(vehicle in label_lower for vehicle in self.config.VEHICLE_CLASSES):
                return self._safe_get_color('vehicle', (255, 0, 0))
            elif any(obj in label_lower for obj in getattr(self.config, 'CLASSROOM_OBJECTS', [])):
                return self._safe_get_color('classroom', (0, 255, 255))
            else:
                return self._safe_get_color('default', (128, 0, 128))
        except Exception as e:
            return (255, 255, 255)

    def _draw_corners(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, length: int, color: tuple):
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)

    def _get_category_info(self, label: str) -> str:
        label_lower = label.lower()
        if any(obj in label_lower for obj in self.config.HUMAN_ANIMAL_CLASSES):
            if label_lower == 'person':
                return "BIOLOGICAL ENTITY"
            elif label_lower in ['cat', 'dog']:
                return "DOMESTIC ANIMAL"
            elif label_lower == 'bird':
                return "AVIAN SPECIES"
            else:
                return "WILDLIFE"
        elif any(vehicle in label_lower for vehicle in self.config.VEHICLE_CLASSES):
            return "VEHICLE"
        elif any(obj in label_lower for obj in getattr(self.config, 'CLASSROOM_OBJECTS', [])):
            return "OBJECT"
        else:
            return "UNKNOWN ENTITY"

    # Keep existing UI methods
    def _draw_wake_word_alert(self, frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        pulse = (math.sin(self.wake_word_animation * 0.3) + 1) * 0.3 + 0.4
        center_x, center_y = frame_width // 2, frame_height // 2
        for i in range(3):
            radius = 40 + i * 15
            alpha = 1.0 - (i * 0.3)
            color = tuple(int(c * alpha * pulse) for c in self.NEON_SKY_BLUE)
            cv2.circle(frame, (center_x, center_y), radius, color, 2, lineType=cv2.LINE_AA)
        activity_radius = 20
        activity_pulse = (math.sin(self.wake_word_animation * 0.5) + 1) * 0.5 + 0.5
        cv2.circle(frame, (center_x, center_y), int(activity_radius * activity_pulse), 
                  self.NEON_GLOW, -1, lineType=cv2.LINE_AA)
        text = "VOICE ACTIVE"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(frame, text, 
                   (center_x - text_size[0] // 2, center_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.NEON_SKY_BLUE, 1, cv2.LINE_AA)
        return frame
    
    def trigger_wake_word(self):
        self.wake_word_detected = True
        self.wake_word_animation = 0
    
    def draw_focus_info(self, frame: np.ndarray, focused_object: Dict, description: str) -> np.ndarray:
        if focused_object and description:
            x1, y1, x2, y2 = focused_object['bbox']
            center_x = (x1 + x2) // 2
            info_height = 70
            info_width = 450
            info_x = max(10, min(center_x - info_width//2, frame.shape[1] - info_width - 10))
            info_y = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (info_x, info_y), (info_x + info_width, info_y + info_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.rectangle(frame, (info_x, info_y), (info_x + info_width, info_y + info_height), 
                         self.NEON_SKY_BLUE, 2, lineType=cv2.LINE_AA)
            focus_text = f"FOCUS: {focused_object['label'].upper()} ({focused_object['confidence']:.1%})"
            cv2.putText(frame, focus_text, (info_x + 10, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.NEON_BRIGHT, 1, cv2.LINE_AA)
            if len(description) > 70:
                description = description[:67] + "..."
            cv2.putText(frame, description, (info_x + 10, info_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return frame
    
    def draw_ui_overlay(self, frame: np.ndarray, object_count: int, fps: int, 
                       focused_object: Optional[Dict]) -> np.ndarray:
        frame_height, frame_width = frame.shape[:2]
        header = np.zeros((45, frame_width, 3), dtype=np.uint8)
        frame[0:45, 0:frame_width] = cv2.addWeighted(frame[0:45, 0:frame_width], 0.8, header, 0.2, 0)
        title = "JARVIS VISUAL INTERFACE - MULTI OBJECT TRACKING"
        cv2.putText(frame, title, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.NEON_SKY_BLUE, 1, cv2.LINE_AA)
        status_text = f"OBJECTS: {object_count} | FPS: {fps} | VOICE: ACTIVE"
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, status_text, (frame_width - status_size[0] - 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.NEON_BRIGHT, 1, cv2.LINE_AA)
        footer_height = 30
        footer = np.zeros((footer_height, frame_width, 3), dtype=np.uint8)
        frame[frame_height-footer_height:frame_height, 0:frame_width] = cv2.addWeighted(
            frame[frame_height-footer_height:frame_height, 0:frame_width], 0.8, footer, 0.2, 0
        )
        instructions = "SAY 'SIDRA' FOR VOICE CONTROL | SPACE: MANUAL VOICE | Q: EXIT"
        instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, instructions, ((frame_width - instr_size[0]) // 2, frame_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.NEON_BRIGHT, 1, cv2.LINE_AA)
        return frame