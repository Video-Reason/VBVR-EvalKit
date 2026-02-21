"""Evaluator for O-61_symbol_edit_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-61_symbol_edit_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class SymbolEditConstraintEvaluator(BaseEvaluator):
    """
    O-61: Symbol edit with constraint evaluator.
    
    Rule-based evaluation:
    - Original preservation (45%): All original symbols remain unchanged
    - Insertion occurred (35%): Correct number of symbols inserted
    - Count correctness (15%): Final count matches expected
    - Layout accuracy (5%): Proper spacing
    """
    
    TASK_WEIGHTS = {
        'original_preservation': 0.45,
        'insertion_occurred': 0.35,
        'count_correctness': 0.15,
        'layout_accuracy': 0.05
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
        
        first_symbols = self._detect_symbols(first_frame)
        final_symbols = self._detect_symbols(final_frame)
        
        first_count = len(first_symbols)
        final_count = len(final_symbols)
        
        # Get expected count from GT if available
        if gt_final_frame is not None:
            gt_final_symbols = self._detect_symbols(gt_final_frame)
            expected_count = len(gt_final_symbols)
        else:
            expected_count = first_count + 2  # Default: insert 2
        
        expected_inserted = expected_count - first_count
        
        # CRITICAL CHECK 1: Symbols must be inserted (not deleted)
        if final_count <= first_count:
            self._last_task_details = {
                'original_preservation': 0.0,
                'insertion_occurred': 0.0,
                'count_correctness': 0.0,
                'layout_accuracy': 0.0,
                'no_insertion': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        # CRITICAL CHECK 2: All original symbols must be preserved
        scores['original_preservation'] = self._evaluate_original_preservation(
            first_symbols, final_symbols
        )
        
        # If original symbols are not preserved, heavily penalize
        if scores['original_preservation'] < 0.5:
            self._last_task_details = {
                'original_preservation': scores['original_preservation'],
                'insertion_occurred': 0.0,
                'count_correctness': 0.0,
                'layout_accuracy': 0.0,
                'originals_changed': True
            }
            return scores['original_preservation'] * self.TASK_WEIGHTS['original_preservation']
        
        # Check if correct number of symbols were inserted
        actual_inserted = final_count - first_count
        if actual_inserted == expected_inserted:
            scores['insertion_occurred'] = 1.0
        elif actual_inserted > 0:
            scores['insertion_occurred'] = max(0.3, 1.0 - abs(actual_inserted - expected_inserted) * 0.2)
        else:
            scores['insertion_occurred'] = 0.0
        
        # CRITICAL: Check if inserted symbols are of the correct type
        # Get GT final symbols to determine target symbol type
        if gt_final_frame is not None:
            gt_final_symbols = self._detect_symbols(gt_final_frame)
            # Find the target symbol type (the one that was duplicated)
            target_hue = self._find_target_symbol_hue(first_symbols, gt_final_symbols)
            
            if target_hue is not None:
                # Check if new symbols in final match the target type
                new_symbols_correct = self._check_new_symbols_type(
                    first_symbols, final_symbols, target_hue
                )
                # Penalize if new symbols don't match target type
                scores['insertion_occurred'] *= new_symbols_correct
        
        # Check count correctness
        if final_count == expected_count:
            scores['count_correctness'] = 1.0
        else:
            scores['count_correctness'] = max(0.0, 1.0 - abs(final_count - expected_count) * 0.2)
        
        scores['layout_accuracy'] = self._evaluate_layout(final_symbols)
        
        self._last_task_details = scores
        self._last_task_details['first_count'] = first_count
        self._last_task_details['final_count'] = final_count
        self._last_task_details['expected_count'] = expected_count
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get average color using mask
            mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
            
            # Get HSV hue for color matching
            color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
            hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
            
            symbols.append({
                'center': (cx, cy),
                'area': area,
                'color': mean_color,
                'hue': int(hsv_c[0]),
                'saturation': int(hsv_c[1])
            })
        
        symbols.sort(key=lambda s: s['center'][0])
        return symbols
    
    def _evaluate_original_preservation(self, first_symbols: List[Dict], 
                                        final_symbols: List[Dict]) -> float:
        """Check if all original symbols are preserved."""
        if len(first_symbols) == 0:
            return 1.0
        
        preserved = 0
        used = set()
        
        for f_sym in first_symbols:
            # Find a matching symbol in final
            best_match = None
            best_score = 0
            
            for i, l_sym in enumerate(final_symbols):
                if i in used:
                    continue
                
                # Check hue match
                hue_diff = abs(f_sym['hue'] - l_sym['hue'])
                hue_diff = min(hue_diff, 180 - hue_diff)
                
                # Check saturation match
                sat_diff = abs(f_sym['saturation'] - l_sym['saturation'])
                
                if hue_diff < 15 and sat_diff < 50:
                    score = 1.0 - hue_diff / 15
                    if score > best_score:
                        best_score = score
                        best_match = i
            
            if best_match is not None:
                preserved += 1
                used.add(best_match)
        
        return preserved / len(first_symbols)
    
    def _evaluate_layout(self, final_symbols: List[Dict]) -> float:
        """Check layout spacing."""
        if len(final_symbols) < 2:
            return 0.5
        
        # Check y-coordinate alignment
        y_coords = [s['center'][1] for s in final_symbols]
        y_var = np.var(y_coords)
        
        if y_var < 100:
            return 1.0
        elif y_var < 500:
            return 0.7
        return 0.2
    
    def _find_target_symbol_hue(self, first_symbols: List[Dict], 
                                  gt_final_symbols: List[Dict]) -> Optional[int]:
        """Find the hue of the target symbol (the one that was duplicated)."""
        # Count hues in first and GT final
        first_hues = {}
        for sym in first_symbols:
            h = sym['hue'] // 10 * 10  # Quantize to 10-degree bins
            first_hues[h] = first_hues.get(h, 0) + 1
        
        gt_hues = {}
        for sym in gt_final_symbols:
            h = sym['hue'] // 10 * 10
            gt_hues[h] = gt_hues.get(h, 0) + 1
        
        # Find hue that increased the most
        max_increase = 0
        target_hue = None
        for h, count in gt_hues.items():
            increase = count - first_hues.get(h, 0)
            if increase > max_increase:
                max_increase = increase
                target_hue = h
        
        return target_hue
    
    def _check_new_symbols_type(self, first_symbols: List[Dict], 
                                final_symbols: List[Dict], 
                                target_hue: int) -> float:
        """Check if new symbols match the target type."""
        # Find symbols in final that are not in first (new symbols)
        first_positions = set()
        for sym in first_symbols:
            first_positions.add((sym['center'][0] // 30, sym['center'][1] // 30))
        
        new_symbols = []
        for sym in final_symbols:
            pos = (sym['center'][0] // 30, sym['center'][1] // 30)
            if pos not in first_positions:
                new_symbols.append(sym)
        
        if len(new_symbols) == 0:
            return 1.0
        
        # Check how many new symbols match the target hue
        correct = 0
        for sym in new_symbols:
            hue_diff = abs(sym['hue'] - target_hue)
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 20:
                correct += 1
        
        return correct / len(new_symbols)
