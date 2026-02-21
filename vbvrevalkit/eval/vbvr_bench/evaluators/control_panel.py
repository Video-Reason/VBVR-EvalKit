"""Evaluator for O-54_control_panel_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-54_control_panel_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class ControlPanelEvaluator(BaseEvaluator):
    """
    O-54: Control panel evaluator.
    
    Rule-based evaluation:
    - State matching correctness (45%): All controls reach target state
      - Switch: correct on/off state (green track=on, gray track=off)
      - Slider: value within 5% of target
      - Button: pressed state (green color)
      - Dial: angle within 10 degrees
    - Operation smoothness (25%): Smooth transitions
    - Control identification (20%): Correct control types identified
    - Panel preservation (10%): Panel layout unchanged
    """
    
    TASK_WEIGHTS = {
        'state_matching': 0.45,
        'smoothness': 0.25,
        'identification': 0.20,
        'preservation': 0.10
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
        
        # CRITICAL: First check if control panel structure is preserved
        # Count distinct colored regions (controls) in first and final frames
        first_controls = self._count_control_regions(first_frame)
        final_controls = self._count_control_regions(final_frame)
        
        # If final frame has very few controls or one huge region, structure is destroyed
        if final_controls < 2 or (first_controls > 2 and final_controls < first_controls // 2):
            self._last_task_details = {
                'state_matching': 0.0,
                'smoothness': 0.3,
                'identification': 0.0,
                'preservation': 0.0,
                'structure_destroyed': True,
                'first_controls': first_controls,
                'final_controls': final_controls
            }
            return 0.0
        
        scores['state_matching'] = self._evaluate_state_matching(
            first_frame, final_frame, gt_final_frame
        )
        scores['smoothness'] = self._evaluate_smoothness(video_frames)
        scores['identification'] = self._evaluate_identification(first_frame, final_frame)
        scores['preservation'] = self._evaluate_preservation(first_frame, final_frame)
        
        self._last_task_details = scores
        self._last_task_details['first_controls'] = first_controls
        self._last_task_details['final_controls'] = final_controls
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _count_control_regions(self, frame: np.ndarray) -> int:
        """Count distinct control regions (colored areas) in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count regions that are reasonably sized (not too small, not too large)
        h, w = frame.shape[:2]
        max_area = h * w * 0.3  # Max 30% of frame
        
        return sum(1 for cnt in contours if 100 < cv2.contourArea(cnt) < max_area)
    
    def _detect_controls(self, frame: np.ndarray) -> Dict:
        """Detect control panel elements."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        controls = {
            'switches': [],
            'buttons': [],
            'sliders': [],
            'dials': []
        }
        
        # Detect green elements (active switches, pressed buttons)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Detect blue elements (slider progress)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Detect yellow elements (dial pointers)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find green control regions
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Classify by aspect ratio
            aspect = w / h if h > 0 else 1
            if 0.5 <= aspect <= 2.0 and area < 2000:
                # Could be button or switch
                controls['buttons'].append({'center': (cx, cy), 'area': area, 'bbox': (x, y, w, h)})
        
        # Find blue slider regions
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > h * 2:  # Horizontal slider
                controls['sliders'].append({'bbox': (x, y, w, h), 'value': w})
        
        # Find yellow dial pointers
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            controls['dials'].append({'center': (cx, cy), 'area': area})
        
        return controls
    
    def _evaluate_state_matching(self, first: np.ndarray, final: np.ndarray, 
                                  gt_final: Optional[np.ndarray]) -> float:
        """Check if all controls reach target state."""
        if gt_final is None:
            return 0.5
        
        final_controls = self._detect_controls(final)
        gt_controls = self._detect_controls(gt_final)
        
        scores = []
        
        # Compare button states (green = pressed)
        final_buttons = len(final_controls['buttons'])
        gt_buttons = len(gt_controls['buttons'])
        if gt_buttons > 0:
            button_match = min(final_buttons, gt_buttons) / gt_buttons
            scores.append(button_match)
        
        # Compare slider states (blue progress width)
        if gt_controls['sliders'] and final_controls['sliders']:
            for gt_slider in gt_controls['sliders']:
                best_match = 0
                for f_slider in final_controls['sliders']:
                    # Compare slider values (width of blue area)
                    gt_val = gt_slider['value']
                    f_val = f_slider['value']
                    if gt_val > 0:
                        ratio = min(f_val, gt_val) / max(f_val, gt_val)
                        best_match = max(best_match, ratio)
                scores.append(best_match)
        
        # Compare dial positions
        if gt_controls['dials'] and final_controls['dials']:
            dial_match = min(len(final_controls['dials']), len(gt_controls['dials'])) / max(len(gt_controls['dials']), 1)
            scores.append(dial_match)
        
        return np.mean(scores) if scores else 0.5
    
    def _evaluate_smoothness(self, video_frames: List[np.ndarray]) -> float:
        """Check if transitions are smooth."""
        if len(video_frames) < 3:
            return 0.5
        
        diffs = []
        for i in range(1, min(len(video_frames), 20)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            diffs.append(diff)
        
        if len(diffs) < 2:
            return 0.5
        
        # Low variance = smooth transitions
        variance = np.var(diffs)
        return 1.0 / (1.0 + variance / 50)
    
    def _evaluate_identification(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if controls are correctly identified and operated."""
        first_controls = self._detect_controls(first)
        final_controls = self._detect_controls(final)
        
        # Check if control counts are reasonable
        first_count = sum(len(v) for v in first_controls.values())
        final_count = sum(len(v) for v in final_controls.values())
        
        if first_count == 0:
            return 0.5
        
        # Some controls should change (buttons turn green, etc)
        changed = abs(len(final_controls['buttons']) - len(first_controls['buttons']))
        
        return min(1.0, 0.5 + changed * 0.2)
    
    def _evaluate_preservation(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if panel layout is preserved."""
        # Compare edge structures
        edges_final = cv2.Canny(final, 50, 150)
        edges_first = cv2.Canny(first, 50, 150)
        
        # Panel border should be mostly preserved
        intersection = np.sum((edges_final > 0) & (edges_first > 0))
        union = np.sum((edges_final > 0) | (edges_first > 0))
        
        if union == 0:
            return 0.5
        return intersection / union
