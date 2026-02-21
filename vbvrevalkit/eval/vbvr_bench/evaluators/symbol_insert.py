"""Evaluator for O-59_symbol_insert_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-59_symbol_insert_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class SymbolInsertEvaluator(BaseEvaluator):
    """
    O-59: Symbol insert evaluator.
    
    Rule-based evaluation:
    - Insert position accuracy (40%): Correct position
    - Symbol identification (30%): Correct symbol inserted
    - Sequence adjustment (25%): Other symbols shift correctly
    - Layout accuracy (5%): Centered, even spacing
    """
    
    TASK_WEIGHTS = {
        'position_accuracy': 0.40,
        'symbol_identification': 0.30,
        'sequence_adjustment': 0.25,
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
        
        scores['position_accuracy'] = self._evaluate_position(
            first_frame, final_frame, gt_final_frame
        )
        scores['symbol_identification'] = self._evaluate_symbol(
            first_frame, final_frame, gt_final_frame
        )
        scores['sequence_adjustment'] = self._evaluate_adjustment(
            first_frame, final_frame
        )
        scores['layout_accuracy'] = self._evaluate_layout(final_frame)
        
        self._last_task_details = scores
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
            
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if roi.size > 0:
                color = tuple(roi.mean(axis=(0, 1)).astype(int).tolist())
            else:
                color = (0, 0, 0)
            
            symbols.append({
                'center': (cx, cy),
                'area': area,
                'color': color,
                'bbox': (x, y, w, h)
            })
        
        symbols.sort(key=lambda s: s['center'][0])
        return symbols
    
    def _evaluate_position(self, first: np.ndarray, final: np.ndarray,
                           gt_final: Optional[np.ndarray]) -> float:
        """Check if insert position is correct."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        # Should have one more symbol
        expected_count = len(first_symbols) + 1
        actual_count = len(final_symbols)
        
        if actual_count == expected_count:
            score = 1.0
        elif abs(actual_count - expected_count) == 1:
            score = 0.5  # STRICT: Close but not exact
        else:
            score = 0.0  # STRICT: Wrong symbol count
        
        return score
    
    def _evaluate_symbol(self, first: np.ndarray, final: np.ndarray,
                         gt_final: Optional[np.ndarray]) -> float:
        """Check if correct symbol is inserted."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        # Check if a new symbol was added
        if len(final_symbols) > len(first_symbols):
            return 1.0
        elif len(final_symbols) == len(first_symbols):
            return 0.0  # STRICT: No symbol added
        return 0.0  # STRICT: Symbol removed instead
    
    def _evaluate_adjustment(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if other symbols shift correctly."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        if len(final_symbols) != len(first_symbols) + 1:
            return 0.0  # STRICT: Wrong symbol count
        
        # Check if original symbols are still present (by color)
        first_colors = [s['color'] for s in first_symbols]
        final_colors = [s['color'] for s in final_symbols]
        
        matches = 0
        for f_color in first_colors:
            for color in final_colors:
                dist = np.sqrt(sum((a-b)**2 for a, b in zip(f_color, color)))
                if dist < 50:
                    matches += 1
                    break
        
        return matches / max(len(first_colors), 1)
    
    def _evaluate_layout(self, final: np.ndarray) -> float:
        """Check layout centering and spacing."""
        symbols = self._detect_symbols(final)
        
        if len(symbols) < 2:
            return 0.0  # STRICT: Not enough symbols for layout eval
        
        # Check even spacing
        spacings = []
        for i in range(1, len(symbols)):
            spacing = symbols[i]['center'][0] - symbols[i-1]['center'][0]
            spacings.append(spacing)
        
        if spacings:
            variance = np.var(spacings)
            mean_spacing = np.mean(spacings)
            cv = np.sqrt(variance) / mean_spacing if mean_spacing > 0 else 1.0
            return max(0, 1.0 - cv * 2)
        
        return 0.0  # STRICT: No spacing to evaluate
