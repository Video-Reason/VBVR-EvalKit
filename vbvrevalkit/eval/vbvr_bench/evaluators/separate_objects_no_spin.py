"""Evaluator for G-24_separate_objects_no_spin_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-24_separate_objects_no_spin_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class SeparateObjectsNoSpinEvaluator(BaseEvaluator):
    """
    G-24: Separate objects (no spin) evaluator.
    
    Rule-based evaluation:
    - Alignment precision (35%): Objects align with dashed target outlines
    - No rotation constraint (30%): Objects maintain original orientation
    - Movement correctness (20%): Horizontal movement to the right
    - Visual fidelity (15%): Shape, color, size preserved
    """
    
    TASK_WEIGHTS = {
        'alignment': 0.35,
        'no_rotation': 0.30,
        'movement': 0.20,
        'fidelity': 0.15
    }
    
    def _detect_colored_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored shapes and their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find non-white/non-black areas
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get bounding rect for orientation
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            
            shapes.append({
                'center': (cx, cy),
                'area': area,
                'angle': angle,
                'contour': cnt
            })
        
        return shapes
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate separate objects no spin task - rule-based, no SSIM."""
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect shapes in initial and final frames
        initial_shapes = self._detect_colored_shapes(first_frame)
        final_shapes = self._detect_colored_shapes(last_frame)
        gt_final_shapes = self._detect_colored_shapes(gt_final_frame) if gt_final_frame is not None else []
        
        # 1. Alignment precision: Compare final positions with GT
        if final_shapes and gt_final_shapes:
            total_dist = 0
            matched = 0
            for fs in final_shapes:
                min_dist = float('inf')
                for gts in gt_final_shapes:
                    dist = safe_distance(fs['center'], gts['center'])
                    min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    total_dist += min_dist
                    matched += 1
            
            if matched > 0:
                avg_dist = total_dist / matched
                # Close match (< 10 pixels) gets full score
                if avg_dist < 10:
                    scores['alignment'] = 1.0
                else:
                    # Lenient threshold (100 pixels)
                    scores['alignment'] = max(0, 1.0 - avg_dist / 100.0)
            else:
                # No matches found - check if shapes moved right
                scores['alignment'] = self._check_rightward_movement(initial_shapes, final_shapes)
        else:
            # Fallback: check if shapes are on right side
            scores['alignment'] = self._check_rightward_movement(initial_shapes, final_shapes)
        
        # 2. No rotation constraint: Compare angles between initial and final
        if initial_shapes and final_shapes:
            angle_diffs = []
            # Sort by area to match shapes (more robust than position)
            initial_sorted = sorted(initial_shapes, key=lambda s: s['area'], reverse=True)
            final_sorted = sorted(final_shapes, key=lambda s: s['area'], reverse=True)
            
            for i, (i_shape, f_shape) in enumerate(zip(initial_sorted, final_sorted)):
                if i >= min(len(initial_sorted), len(final_sorted)):
                    break
                angle_diff = abs(i_shape['angle'] - f_shape['angle'])
                angle_diff = min(angle_diff, 90 - angle_diff)  # Handle angle wrapping
                angle_diffs.append(angle_diff)
            
            if angle_diffs:
                avg_angle_diff = np.mean(angle_diffs)
                # Small angle diff (< 2 degrees) gets full score
                if avg_angle_diff < 2:
                    scores['no_rotation'] = 1.0
                elif avg_angle_diff < 15:
                    scores['no_rotation'] = 0.8
                elif avg_angle_diff < 30:
                    scores['no_rotation'] = 0.2  # Detection failed
                else:
                    scores['no_rotation'] = max(0, 1.0 - avg_angle_diff / 45.0)
            else:
                scores['no_rotation'] = 0.2  # Detection failed
        else:
            scores['no_rotation'] = 0.2  # Detection failed
        
        # 3. Movement correctness: Check horizontal motion
        frame_diff = cv2.absdiff(first_frame, last_frame)
        if np.mean(frame_diff) < 5:  # Very similar frames
            scores['movement'] = 1.0
        else:
            flow_result = compute_optical_flow(first_frame, last_frame)
            if flow_result is not None:
                flow, _ = flow_result
                h_flow = np.abs(flow[:, :, 0]).mean()
                v_flow = np.abs(flow[:, :, 1]).mean()
                if h_flow + v_flow > 0:
                    scores['movement'] = h_flow / (h_flow + v_flow)
                else:
                    scores['movement'] = 0.8
            else:
                scores['movement'] = self._check_horizontal_movement(initial_shapes, final_shapes)
        
        # 4. Fidelity: Check shape preservation
        if initial_shapes and final_shapes:
            # Check count preservation
            count_match = max(0, 1.0 - abs(len(initial_shapes) - len(final_shapes)) / max(len(initial_shapes), 1))
            
            # Check area preservation
            initial_total_area = sum(s['area'] for s in initial_shapes)
            final_total_area = sum(s['area'] for s in final_shapes)
            area_ratio = min(initial_total_area, final_total_area) / max(initial_total_area, final_total_area, 1)
            
            scores['fidelity'] = 0.5 * count_match + 0.5 * area_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _check_rightward_movement(self, initial_shapes: List[Dict], final_shapes: List[Dict]) -> float:
        """Check if shapes moved to the right side."""
        if not initial_shapes or not final_shapes:
            return 0.5
        
        initial_avg_x = np.mean([s['center'][0] for s in initial_shapes])
        final_avg_x = np.mean([s['center'][0] for s in final_shapes])
        
        # Shapes should move right
        if final_avg_x > initial_avg_x:
            return min(1.0, (final_avg_x - initial_avg_x) / 100.0 + 0.5)
        return 0.3
    
    def _check_horizontal_movement(self, initial_shapes: List[Dict], final_shapes: List[Dict]) -> float:
        """Check if movement is primarily horizontal."""
        if not initial_shapes or not final_shapes:
            return 0.5
        
        initial_centers = [s['center'] for s in initial_shapes]
        final_centers = [s['center'] for s in final_shapes]
        
        # Calculate average movement
        if len(initial_centers) != len(final_centers):
            return 0.5
        
        total_h = 0
        total_v = 0
        for ic, fc in zip(initial_centers, final_centers):
            total_h += abs(fc[0] - ic[0])
            total_v += abs(fc[1] - ic[1])
        
        if total_h + total_v > 0:
            return total_h / (total_h + total_v)
        return 0.8
