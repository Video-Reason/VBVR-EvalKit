"""Evaluator for O-37_light_sequence_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-37_light_sequence_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class LightSequenceEvaluator(BaseEvaluator):
    """
    O-37: Light Sequence State Control
    
    Task: Modify light states according to spatial/mathematical rules.
    Lights can be on (gold) or off (gray). 6 rule types.
    
    Key evaluation criteria:
    1. Rule understanding (35%) - Correct interpretation of rule type
    2. Position identification (30%) - Correct light positions identified
    3. State transition (25%) - Correct on/off states
    4. Visual quality (10%) - Colors and glow effects
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'rule_understanding': 0.35,
            'position_identification': 0.30,
            'state_transition': 0.25,
            'visual_quality': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate light sequence state control - RULE-BASED comparison."""
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gen_final = cv2.resize(gen_final, (gt_final.shape[1], gt_final.shape[0]))
        
        scores = {}
        
        # RULE-BASED: Detect light positions from GT first frame, then check states
        # Step 1: Find all light positions from GT first frame
        light_positions = self._detect_all_light_positions(gt_first)
        
        if len(light_positions) == 0:
            # Fallback to pixel comparison if detection fails
            final_diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['rule_understanding'] = 1.0 if final_diff < 10 else 0.0
            scores['position_identification'] = 1.0 if final_diff < 10 else 0.0
            scores['state_transition'] = 1.0 if final_diff < 10 else 0.0
            scores['visual_quality'] = 1.0 if final_diff < 15 else 0.0
            self._last_task_details = scores
            return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
        
        # Step 2: Detect which lights are ON in GT final (expected states)
        gt_on_states = self._get_light_states(gt_final, light_positions)
        
        # Step 3: Detect which lights are ON in generated final
        gen_on_states = self._get_light_states(gen_final, light_positions)
        
        # Step 4: Compare states - STRICT rule-based
        # Count matching states
        matching_states = sum(1 for g, e in zip(gen_on_states, gt_on_states) if g == e)
        total_lights = len(light_positions)
        
        state_accuracy = matching_states / total_lights if total_lights > 0 else 0
        
        # All scores depend on state accuracy
        # If states don't match, the rule was not followed correctly
        if state_accuracy == 1.0:  # Perfect match
            scores['rule_understanding'] = 1.0
            scores['position_identification'] = 1.0
            scores['state_transition'] = 1.0
            scores['visual_quality'] = 1.0
        elif state_accuracy >= 0.8:  # Minor errors
            scores['rule_understanding'] = 0.5
            scores['position_identification'] = 0.5
            scores['state_transition'] = 0.5
            scores['visual_quality'] = 0.8
        else:  # Wrong states
            scores['rule_understanding'] = 0.0
            scores['position_identification'] = 0.0
            scores['state_transition'] = 0.0
            scores['visual_quality'] = 0.3
        
        self._last_task_details = scores
        self._last_task_details['gt_on_states'] = str(gt_on_states)
        self._last_task_details['gen_on_states'] = str(gen_on_states)
        self._last_task_details['state_accuracy'] = state_accuracy
        
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_all_light_positions(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect all light positions (both on and off) from the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find non-white regions (lights are colored, background is white)
        non_white = (gray < 250).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        positions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter noise
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    positions.append((cx, cy))
        
        # Sort by x position (left to right)
        positions.sort(key=lambda p: p[0])
        return positions
    
    def _get_light_states(self, frame: np.ndarray, positions: List[Tuple[int, int]]) -> List[bool]:
        """Get ON/OFF state for each light position."""
        states = []
        for cx, cy in positions:
            # Sample color at light center
            y1, y2 = max(0, cy-10), min(frame.shape[0], cy+10)
            x1, x2 = max(0, cx-10), min(frame.shape[1], cx+10)
            region = frame[y1:y2, x1:x2]
            
            if region.size > 0:
                mean_color = np.mean(region, axis=(0, 1))
                b, g, r = mean_color
                # Gold/yellow: high R, high G, low B
                is_on = r > 180 and g > 100 and b < 150
                states.append(is_on)
            else:
                states.append(False)
        
        return states
    
    def _detect_light_states(self, frame: np.ndarray) -> List[Dict]:
        """Detect lights and their on/off states using non-white region detection."""
        lights = []
        
        # Find all non-white regions (lights are colored, background is white)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        non_white = (gray < 250).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Skip very small regions
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get average color in the region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]
            
            b_val, g_val, r_val = mean_color[0], mean_color[1], mean_color[2]
            
            # Determine if ON (gold/orange/yellow) or OFF (gray)
            # Gold/orange: high R, high G, low B (BGR format)
            # Gray: similar R, G, B values with low saturation
            color_diff = max(abs(r_val - g_val), abs(g_val - b_val), abs(r_val - b_val))
            
            # ON lights have high color difference (gold/orange is saturated)
            # OFF lights have low color difference (gray is desaturated)
            is_on = (r_val > 200 and g_val > 150 and b_val < 100) or \
                    (r_val > 220 and g_val > 180) or \
                    (color_diff > 50 and r_val > 180)  # Saturated warm color
            
            lights.append({
                'center': (cx, cy),
                'area': area,
                'is_on': is_on,
                'color': mean_color
            })
        
        # Sort by x position (left to right)
        lights.sort(key=lambda l: l['center'][0])
        return lights
    
    def _detect_lights_by_color(self, frame: np.ndarray) -> List[Dict]:
        """Fallback detection by color."""
        lights = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Gold/yellow/orange detection - expanded range
        # Hue: 10-45 for yellow/gold/orange
        lower_gold = np.array([10, 80, 120])
        upper_gold = np.array([45, 255, 255])
        gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Also detect by RGB for gold colors that may not be well captured in HSV
        # Gold: RGB(255,215,0), RGB(255,165,0)
        b, g, r = cv2.split(frame)
        rgb_gold_mask = ((r > 180) & (g > 100) & (b < 150)).astype(np.uint8) * 255
        gold_mask = cv2.bitwise_or(gold_mask, rgb_gold_mask)
        
        # Gray detection
        lower_gray = np.array([0, 0, 60])
        upper_gray = np.array([180, 60, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Find gold lights (ON)
        contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lower threshold
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    lights.append({'center': (cx, cy), 'is_on': True, 'area': area})
        
        # Find gray lights (OFF)
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Lower threshold
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:  # More lenient circularity
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            # Check not already added as gold
                            is_new = all(abs(l['center'][0] - cx) > 15 for l in lights)
                            if is_new:
                                lights.append({'center': (cx, cy), 'is_on': False, 'area': area})
        
        lights.sort(key=lambda l: l['center'][0])
        return lights
    
    def _evaluate_rule_understanding(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate if the rule was correctly understood and applied."""
        if not gt_lights:
            return 0.5
        
        # Get on/off pattern
        gt_pattern = [l['is_on'] for l in gt_lights]
        gen_pattern = [l['is_on'] for l in gen_lights]
        
        if len(gen_pattern) != len(gt_pattern):
            # Different number of lights
            return max(0.0, max(0, 1.0 - abs(len(gen_pattern) - len(gt_pattern)) / len(gt_pattern)))
        
        # Compare patterns
        matches = sum(1 for g, gt in zip(gen_pattern, gt_pattern) if g == gt)
        return matches / len(gt_pattern)
    
    def _evaluate_position_identification(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate if lights are in correct positions."""
        if not gt_lights or not gen_lights:
            return 0.0 if not gen_lights else 0.5
        
        # Compare positions
        position_scores = []
        for gt_l in gt_lights:
            best_dist = float('inf')
            for gen_l in gen_lights:
                dist = safe_distance(gt_l['center'], gen_l['center'])
                best_dist = min(best_dist, dist)
            
            if best_dist < 20:
                position_scores.append(1.0)
            elif best_dist < 50:
                position_scores.append(0.7)
            else:
                position_scores.append(max(0.2, 1.0 - best_dist / 100))
        
        return np.mean(position_scores) if position_scores else 0.0
    
    def _evaluate_state_transition(self, gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate on/off state accuracy."""
        if not gt_lights or not gen_lights:
            return 0.0
        
        # Match lights by position and compare states
        correct_states = 0
        total = len(gt_lights)
        
        for gt_l in gt_lights:
            # Find closest generated light
            best_match = None
            best_dist = float('inf')
            
            for gen_l in gen_lights:
                dist = safe_distance(gt_l['center'], gen_l['center'])
                if dist < best_dist:
                    best_dist = dist
                    best_match = gen_l
            
            if best_match and best_dist < 50:
                if best_match['is_on'] == gt_l['is_on']:
                    correct_states += 1
        
        return correct_states / total if total > 0 else 0.0
    
    def _evaluate_light_visual_quality(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                                        gen_lights: List[Dict], gt_lights: List[Dict]) -> float:
        """Evaluate visual quality of lights (renamed to avoid conflict with base class)."""
        if not gen_lights:
            return 0.0
        
        # Check if ON lights have gold/yellow color
        on_lights = [l for l in gen_lights if l.get('is_on', False)]
        off_lights = [l for l in gen_lights if not l.get('is_on', False)]
        
        quality_scores = []
        
        # Check ON lights are gold/yellow
        for light in on_lights:
            if 'color' in light:
                r, g, b = light['color'][2], light['color'][1], light['color'][0]
                # Gold should be high R, high G, low B
                if r > 180 and g > 150 and b < 150:
                    quality_scores.append(1.0)
                elif r > 150 and g > 120:
                    quality_scores.append(0.7)
                else:
                    quality_scores.append(0.3)
        
        return np.mean(quality_scores) if quality_scores else 0.5
