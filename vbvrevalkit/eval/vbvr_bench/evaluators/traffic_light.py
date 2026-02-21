"""Evaluator for O-52_traffic_light_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-52_traffic_light_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class TrafficLightEvaluator(BaseEvaluator):
    """
    O-52: Traffic Light Reasoning
    
    Task: Understand two opposite traffic lights' state switching rules 
    and countdown logic, predict final state after countdown.
    
    Key evaluation criteria:
    1. Final state accuracy (35%) - Correct light colors
    2. Countdown correctness (30%) - Proper decrement
    3. Switch timing (25%) - Switch at countdown=0
    4. Opposite rule (10%) - Lights always opposite
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'final_state_accuracy': 0.35,
            'countdown_correctness': 0.30,
            'switch_timing': 0.25,
            'opposite_rule': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate traffic light state reasoning."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # 1. Final state accuracy (35%): Check light colors
        final_score = self._evaluate_final_state(gen_final, gt_final)
        scores['final_state_accuracy'] = final_score
        
        # 2. Countdown correctness (30%): Track countdown through video
        countdown_score = self._evaluate_countdown(video_frames)
        scores['countdown_correctness'] = countdown_score
        
        # 3. Switch timing (25%): Check if switch happens at right time
        timing_score = self._evaluate_switch_timing(video_frames)
        scores['switch_timing'] = timing_score
        
        # 4. Opposite rule (10%): Check if lights are opposite
        opposite_score = self._evaluate_opposite_rule(gen_final)
        scores['opposite_rule'] = opposite_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_traffic_lights(self, frame: np.ndarray) -> Dict:
        """Detect traffic light colors."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Split into left and right halves
        left_half = hsv[:, :w//2]
        right_half = hsv[:, w//2:]
        
        def detect_color(region):
            # Red detection
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask = cv2.inRange(region, lower_red1, upper_red1) | \
                       cv2.inRange(region, lower_red2, upper_red2)
            
            # Green detection
            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(region, lower_green, upper_green)
            
            red_pixels = np.sum(red_mask > 0)
            green_pixels = np.sum(green_mask > 0)
            
            if red_pixels > green_pixels and red_pixels > 100:
                return 'red'
            elif green_pixels > red_pixels and green_pixels > 100:
                return 'green'
            else:
                return 'unknown'
        
        return {
            'left': detect_color(left_half),
            'right': detect_color(right_half)
        }
    
    def _evaluate_final_state(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if final light colors are correct."""
        gen_lights = self._detect_traffic_lights(gen_frame)
        gt_lights = self._detect_traffic_lights(gt_frame)
        
        matches = 0
        total = 0
        
        for side in ['left', 'right']:
            if gt_lights[side] != 'unknown':
                total += 1
                if gen_lights[side] == gt_lights[side]:
                    matches += 1
        
        # STRICT: If no lights detected or no match, return 0
        return matches / total if total > 0 else 0.0
    
    def _evaluate_countdown(self, frames: List[np.ndarray]) -> float:
        """Evaluate countdown behavior through video."""
        if len(frames) < 5:
            return 0.0  # STRICT: Not enough frames
        
        # Track light colors through video
        colors_over_time = []
        for frame in frames[::max(1, len(frames)//10)]:
            lights = self._detect_traffic_lights(frame)
            colors_over_time.append(lights)
        
        # Check if there's a state change (indicating countdown reached 0)
        changes = 0
        for i in range(1, len(colors_over_time)):
            if colors_over_time[i]['left'] != colors_over_time[i-1]['left']:
                changes += 1
        
        # Should have 0 or 1 change typically
        if changes <= 1:
            return 1.0
        elif changes <= 2:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_switch_timing(self, frames: List[np.ndarray]) -> float:
        """Evaluate if state switch happens at appropriate time."""
        if len(frames) < 3:
            return 0.0  # STRICT: Not enough frames
        
        # Find when color change happens
        first_lights = self._detect_traffic_lights(frames[0])
        last_lights = self._detect_traffic_lights(frames[-1])
        
        # Check if there was a change
        changed = first_lights['left'] != last_lights['left'] or \
                  first_lights['right'] != last_lights['right']
        
        if changed:
            # Find when change happened
            for i, frame in enumerate(frames):
                lights = self._detect_traffic_lights(frame)
                if lights['left'] != first_lights['left']:
                    # Change happened at frame i
                    # Should be towards the end (after countdown)
                    progress = i / len(frames)
                    if progress > 0.5:
                        return 1.0
                    elif progress > 0.3:
                        return 0.7
                    else:
                        return 0.4
        
        return 0.0  # STRICT: No timing change detected
    
    def _evaluate_opposite_rule(self, frame: np.ndarray) -> float:
        """Evaluate if lights are opposite (one red, one green)."""
        lights = self._detect_traffic_lights(frame)
        
        left = lights['left']
        right = lights['right']
        
        # Should be opposite
        if (left == 'red' and right == 'green') or (left == 'green' and right == 'red'):
            return 1.0
        elif left == 'unknown' or right == 'unknown':
            return 0.0  # STRICT: Cannot detect lights
        else:
            return 0.0  # Same color - violation
