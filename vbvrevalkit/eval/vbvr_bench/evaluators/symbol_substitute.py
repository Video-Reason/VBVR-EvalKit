"""Evaluator for O-60_symbol_substitute_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-60_symbol_substitute_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class SymbolSubstituteEvaluator(BaseEvaluator):
    """
    O-60: Symbol substitute evaluator.
    
    Rule-based evaluation:
    - Symbol count preservation (40%): Same number of symbols
    - Symbol preservation (35%): All OTHER symbols' colors unchanged
    - Substitution occurred (20%): Exactly one symbol changed
    - Animation quality (5%): Smooth cross-fade
    """
    
    TASK_WEIGHTS = {
        'count_preservation': 0.40,
        'symbol_preservation': 0.35,
        'substitution_occurred': 0.20,
        'animation_quality': 0.05
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
        
        # CRITICAL CHECK 1: Symbol count must remain the same
        if final_count != first_count:
            self._last_task_details = {
                'count_preservation': 0.0,
                'symbol_preservation': 0.0,
                'substitution_occurred': 0.0,
                'animation_quality': 0.0,
                'count_mismatch': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        scores['count_preservation'] = 1.0
        
        # CRITICAL CHECK 2: All other symbols' colors must remain unchanged
        # Match symbols by position and check color preservation
        changed_count, preservation_score = self._evaluate_symbol_changes(
            first_symbols, final_symbols
        )
        scores['symbol_preservation'] = preservation_score
        
        # CRITICAL CHECK 3: Exactly one symbol must have changed
        if changed_count == 1:
            scores['substitution_occurred'] = 1.0
        elif changed_count == 0:
            scores['substitution_occurred'] = 0.0  # No substitution
        else:
            scores['substitution_occurred'] = max(0.0, 1.0 - (changed_count - 1) * 0.3)
        
        scores['animation_quality'] = self._evaluate_animation(video_frames)
        
        self._last_task_details = scores
        self._last_task_details['changed_count'] = changed_count
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
    
    def _evaluate_symbol_changes(self, first_symbols: List[Dict], 
                                  final_symbols: List[Dict]) -> Tuple[int, float]:
        """Count how many symbols changed color and calculate preservation score."""
        if len(first_symbols) != len(final_symbols):
            return len(first_symbols), 0.0
        
        changed_count = 0
        preserved_count = 0
        
        # Match by position (sorted by x)
        for f_sym, l_sym in zip(first_symbols, final_symbols):
            # Check position match
            pos_dist = abs(f_sym['center'][0] - l_sym['center'][0])
            if pos_dist > 50:
                # Position shifted too much - treat as changed
                changed_count += 1
                continue
            
            # Check color match using hue
            hue_diff = abs(f_sym['hue'] - l_sym['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)
            
            # Also check saturation for white/gray symbols
            sat_diff = abs(f_sym['saturation'] - l_sym['saturation'])
            
            if hue_diff < 15 and sat_diff < 50:
                preserved_count += 1
            else:
                changed_count += 1
        
        # Preservation score: (n-1) symbols should be preserved (one is substituted)
        expected_preserved = len(first_symbols) - 1
        preservation_score = preserved_count / expected_preserved if expected_preserved > 0 else 0.0
        
        return changed_count, preservation_score
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Check if cross-fade animation is smooth."""
        if len(video_frames) < 3:
            return 0.5
        
        changes = []
        for i in range(1, len(video_frames)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            changes.append(diff)
        
        if not changes:
            return 0.5
        
        variance = np.var(changes)
        return 1.0 / (1.0 + variance / 50)
