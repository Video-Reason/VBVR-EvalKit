"""Evaluator for O-5_symbol_deletion_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-5_symbol_deletion_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class SymbolDeletionEvaluator(BaseEvaluator):
    """
    O-5: Symbol deletion evaluator.
    
    Rule-based evaluation:
    - Target identification & deletion (40%): Correct symbol (with red border) removed
    - Symbol preservation (35%): All OTHER symbols' colors remain unchanged
    - Sequence order preservation (15%): Remaining symbols in order
    - Layout & alignment (10%): Centered, evenly spaced
    """
    
    TASK_WEIGHTS = {
        'deletion_accuracy': 0.40,
        'symbol_preservation': 0.35,
        'order_preservation': 0.15,
        'layout_alignment': 0.10
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
        
        # Get symbols with colors
        first_symbols = self._detect_symbols_with_color(first_frame)
        final_symbols = self._detect_symbols_with_color(final_frame)
        
        first_count = len(first_symbols)
        final_count = len(final_symbols)
        
        # CRITICAL CHECK 1: Exactly one symbol must be deleted
        if final_count != first_count - 1:
            # Task failed - wrong number of symbols deleted
            if final_count >= first_count:
                # No deletion or symbols added
                self._last_task_details = {
                    'deletion_accuracy': 0.0,
                    'symbol_preservation': 0.0,
                    'order_preservation': 0.0,
                    'layout_alignment': 0.0,
                    'no_deletion': True
                }
                return 0.0
            elif final_count < first_count - 1:
                # Too many deleted
                self._last_task_details = {
                    'deletion_accuracy': 0.1,
                    'symbol_preservation': 0.0,
                    'order_preservation': 0.0,
                    'layout_alignment': 0.0,
                    'too_many_deleted': True
                }
                return 0.04  # 0.1 * 0.4
        
        # Check deletion accuracy (was the correct symbol deleted?)
        scores['deletion_accuracy'] = self._evaluate_deletion(first_symbols, final_symbols, first_frame)
        
        # CRITICAL: Check if other symbols' colors are preserved
        scores['symbol_preservation'] = self._evaluate_symbol_preservation(first_symbols, final_symbols, first_frame)
        
        # If symbol preservation is very low, it means colors changed significantly
        if scores['symbol_preservation'] < 0.5:
            scores['order_preservation'] = 0.0
            scores['layout_alignment'] = 0.0
            self._last_task_details = scores
            self._last_task_details['colors_changed'] = True
            return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
        
        scores['order_preservation'] = self._evaluate_order(first_symbols, final_symbols)
        scores['layout_alignment'] = self._evaluate_layout(final_symbols)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_symbols_with_color(self, frame: np.ndarray) -> List[Dict]:
        """Detect symbols with their centers and average colors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 20000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    # Get HSV hue for color matching
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    symbols.append({
                        'center': (cx, cy),
                        'color': mean_color,
                        'hue': int(hsv[0]),
                        'area': area
                    })
        
        return sorted(symbols, key=lambda s: s['center'][0])  # Sort by x
    
    def _detect_red_border(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the red border marking the target symbol."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        return None
    
    def _evaluate_deletion(self, first_symbols: List[Dict], final_symbols: List[Dict], first_frame: np.ndarray) -> float:
        """Check if exactly one symbol is deleted (the one with red border)."""
        first_count = len(first_symbols)
        final_count = len(final_symbols)
        
        if final_count != first_count - 1:
            if final_count == first_count:
                return 0.2  # Nothing deleted
            elif final_count < first_count - 1:
                return 0.1  # Too many deleted
            else:
                return 0.0  # Symbols added
        
        # Check if the deleted symbol was the one with red border
        red_border_pos = self._detect_red_border(first_frame)
        if red_border_pos is None:
            return 0.7  # Can't verify which was marked
        
        # Find which symbol was deleted
        first_centers = [s['center'] for s in first_symbols]
        final_centers = [s['center'] for s in final_symbols]
        
        # Find the deleted symbol by checking which first symbol is missing
        deleted_idx = None
        for i, fc in enumerate(first_centers):
            # Check if this symbol has a match in final
            has_match = False
            for fnc in final_centers:
                # Allow some position shift due to re-centering
                if abs(fc[0] - fnc[0]) < 100 and abs(fc[1] - fnc[1]) < 50:
                    has_match = True
                    break
            if not has_match:
                deleted_idx = i
                break
        
        if deleted_idx is None:
            return 0.5
        
        # Check if deleted symbol was near the red border
        deleted_center = first_centers[deleted_idx]
        dist_to_red = np.sqrt((deleted_center[0] - red_border_pos[0])**2 + 
                              (deleted_center[1] - red_border_pos[1])**2)
        
        if dist_to_red < 50:
            return 1.0  # Correct symbol deleted
        elif dist_to_red < 100:
            return 0.6
        else:
            return 0.3  # Wrong symbol deleted
    
    def _evaluate_symbol_preservation(self, first_symbols: List[Dict], final_symbols: List[Dict], first_frame: np.ndarray) -> float:
        """CRITICAL: Check if all OTHER symbols' colors remain unchanged."""
        if len(final_symbols) == 0:
            return 0.0
        
        # Find which symbol was marked for deletion (has red border)
        red_border_pos = self._detect_red_border(first_frame)
        
        # Get colors of non-deleted symbols from first frame
        expected_symbols = []
        for sym in first_symbols:
            # Skip the marked symbol
            if red_border_pos:
                dist = np.sqrt((sym['center'][0] - red_border_pos[0])**2 + 
                              (sym['center'][1] - red_border_pos[1])**2)
                if dist < 50:
                    continue
            expected_symbols.append(sym)
        
        if len(expected_symbols) == 0 or len(final_symbols) == 0:
            return 0.5
        
        # STRICT: Each expected symbol must have EXACTLY one matching symbol in final
        # Use greedy matching with strict hue threshold
        matched = 0
        used = set()
        for exp_sym in expected_symbols:
            exp_hue = exp_sym['hue']
            best_match = None
            best_diff = float('inf')
            for i, act_sym in enumerate(final_symbols):
                if i in used:
                    continue
                act_hue = act_sym['hue']
                
                # Calculate hue difference with proper wrapping
                hue_diff = abs(exp_hue - act_hue)
                hue_diff = min(hue_diff, 180 - hue_diff)
                
                if hue_diff < best_diff:
                    best_diff = hue_diff
                    best_match = i
            
            # STRICT threshold: hue must be within 15 degrees
            if best_match is not None and best_diff < 15:
                matched += 1
                used.add(best_match)
        
        # If not all symbols are matched, return low score
        preservation_ratio = matched / len(expected_symbols) if expected_symbols else 0.0
        
        # If less than all expected symbols are preserved, heavily penalize
        if preservation_ratio < 1.0:
            return preservation_ratio * 0.5  # Scale down
        
        return 1.0
    
    def _evaluate_order(self, first_symbols: List[Dict], final_symbols: List[Dict]) -> float:
        """Check if remaining symbols maintain order."""
        if len(final_symbols) < 2:
            return 0.5
        
        # Check if x-coordinates maintain relative order
        final_x = [s['center'][0] for s in final_symbols]
        is_ordered = all(final_x[i] < final_x[i+1] for i in range(len(final_x)-1))
        
        return 1.0 if is_ordered else 0.5
    
    def _evaluate_layout(self, final_symbols: List[Dict]) -> float:
        """Check horizontal alignment."""
        if len(final_symbols) < 2:
            return 0.5
        
        y_coords = [s['center'][1] for s in final_symbols]
        y_var = np.var(y_coords)
        
        if y_var < 100:
            return 1.0
        elif y_var < 500:
            return 0.7
        else:
            return 0.4
