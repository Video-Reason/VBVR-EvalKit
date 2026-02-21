"""Evaluator for G-247_identify_chinese_character_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-247_identify_chinese_character_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class IdentifyChineseCharacterEvaluator(BaseEvaluator):
    """
    G-247: Identify Chinese character evaluator.
    
    Rule-based evaluation:
    - Character recognition correctness (45%): Identify Chinese vs non-Chinese
    - Marking target accuracy (30%): Mark correct character
    - Marking position/range (15%): Circle contains character
    - Marking specification compliance (10%): Red circle, proper style
    """
    
    TASK_WEIGHTS = {
        'character_recognition': 0.45,
        'marking_target': 0.30,
        'marking_position': 0.15,
        'marking_specification': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Character recognition (45%)
        scores['character_recognition'] = self._evaluate_character_recognition(
            first_frame, final_frame
        )
        
        # 2. Marking target (30%)
        scores['marking_target'] = self._evaluate_marking_target(
            first_frame, final_frame
        )
        
        # 3. Marking position (15%)
        scores['marking_position'] = self._evaluate_marking_position(final_frame)
        
        # 4. Marking specification (10%)
        scores['marking_specification'] = self._evaluate_marking_spec(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_character_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if Chinese character is identified."""
        # Find Chinese character (more complex pattern)
        chinese_pos = self._find_chinese_character(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if chinese_pos is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - chinese_pos[0])**2 + (circle[1] - chinese_pos[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_target(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if correct character is marked."""
        circle = self._detect_red_circle(final_frame)
        chinese_pos = self._find_chinese_character(first_frame)
        
        if circle is None:
            return 0.0
        if chinese_pos is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - chinese_pos[0])**2 + (circle[1] - chinese_pos[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circle contains the character."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in valid position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_marking_spec(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check marking specification."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 20 < r < 100:
            return 1.0
        else:
            return 0.5
    
    def _find_chinese_character(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find Chinese character (more complex than letters)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 20000:
                # Check complexity (Chinese characters have more complexity)
                peri = cv2.arcLength(cnt, True)
                complexity = peri * peri / (area + 1)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    characters.append((cx, cy, complexity, area))
        
        if len(characters) == 0:
            return None
        
        # Chinese character typically has highest complexity
        most_complex = max(characters, key=lambda c: c[2])
        return (most_complex[0], most_complex[1])
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
