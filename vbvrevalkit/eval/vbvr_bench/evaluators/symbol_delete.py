"""Evaluator for O-58_symbol_delete_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-58_symbol_delete_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class SymbolDeleteEvaluator(BaseEvaluator):
    """
    O-58: Symbol delete evaluator.
    
    Rule-based evaluation:
    - Target identification & deletion (40%): Correct symbol removed
    - Sequence reorganization (30%): Remaining symbols shift left
    - Order preservation (20%): Relative order maintained
    - Symbol fidelity (10%): Remaining symbols unchanged
    """
    
    TASK_WEIGHTS = {
        'deletion_accuracy': 0.40,
        'reorganization': 0.30,
        'order_preservation': 0.20,
        'symbol_fidelity': 0.10
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
        
        scores['deletion_accuracy'] = self._evaluate_deletion(
            first_frame, final_frame, gt_final_frame
        )
        scores['reorganization'] = self._evaluate_reorganization(
            first_frame, final_frame
        )
        scores['order_preservation'] = self._evaluate_order(
            first_frame, final_frame
        )
        scores['symbol_fidelity'] = self._evaluate_fidelity(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_symbols(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored symbols in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find colored (non-white, non-black) regions
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
            
            # Get dominant color
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
        
        # Sort by x position
        symbols.sort(key=lambda s: s['center'][0])
        return symbols
    
    def _evaluate_deletion(self, first: np.ndarray, final: np.ndarray,
                           gt_final: Optional[np.ndarray]) -> float:
        """Check if correct symbol is deleted."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        # Should have one fewer symbol
        expected_count = len(first_symbols) - 1
        actual_count = len(final_symbols)
        
        if actual_count == expected_count:
            score = 1.0
        elif actual_count == expected_count - 1 or actual_count == expected_count + 1:
            score = 0.5  # STRICT: Close but not exact
        else:
            score = 0.0  # STRICT: Wrong symbol count
        
        return score
    
    def _evaluate_reorganization(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if symbols shift left correctly."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        if not final_symbols:
            return 0.0
        
        # Check if symbols are evenly spaced
        if len(final_symbols) >= 2:
            spacings = []
            for i in range(1, len(final_symbols)):
                spacing = final_symbols[i]['center'][0] - final_symbols[i-1]['center'][0]
                spacings.append(spacing)
            
            if spacings:
                variance = np.var(spacings)
                mean_spacing = np.mean(spacings)
                cv = np.sqrt(variance) / mean_spacing if mean_spacing > 0 else 1.0
                return max(0, 1.0 - cv)
        
        return 0.0  # STRICT: Not enough symbols to evaluate reorganization
    
    def _evaluate_order(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if relative order is maintained."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        if not final_symbols:
            return 0.0
        
        # Compare colors of remaining symbols (should be subset of first)
        first_colors = [s['color'] for s in first_symbols]
        final_colors = [s['color'] for s in final_symbols]
        
        # Check if colors maintain relative order
        matches = 0
        for i, f_color in enumerate(final_colors):
            # Find matching color in first
            for j, color in enumerate(first_colors):
                dist = np.sqrt(sum((a-b)**2 for a, b in zip(f_color, color)))
                if dist < 50:  # Close enough
                    matches += 1
                    break
        
        return matches / max(len(final_colors), 1)
    
    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if remaining symbols are unchanged."""
        first_symbols = self._detect_symbols(first)
        final_symbols = self._detect_symbols(final)
        
        if not final_symbols:
            return 0.0
        
        # Check area preservation
        first_areas = sorted([s['area'] for s in first_symbols])
        final_areas = sorted([s['area'] for s in final_symbols])
        
        if len(first_areas) > len(final_areas):
            # Remove one area (the deleted symbol)
            first_areas_subset = first_areas[:-1] if len(first_areas) > 0 else []
        else:
            first_areas_subset = first_areas
        
        if len(first_areas_subset) == len(final_areas):
            area_diffs = [abs(a - b) / max(a, 1) for a, b in zip(first_areas_subset, final_areas)]
            return 1.0 - min(1.0, np.mean(area_diffs))
        
        return 0.0  # STRICT: Symbol counts don't match
