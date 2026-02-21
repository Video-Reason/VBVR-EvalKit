"""Evaluator for O-55_rotation_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-55_rotation_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class RotationEvaluator(BaseEvaluator):
    """
    O-55: 3D Mental Rotation
    
    Task: Given 3D voxel structure, show view after camera rotates 
    180° horizontally (keeping elevation constant).
    
    Key evaluation criteria:
    1. 3D spatial understanding (35%) - Correct structure comprehension
    2. Rotation angle accuracy (35%) - Exactly 180° rotation
    3. View consistency (20%) - Same structure, opposite view
    4. Rendering quality (10%) - Proper 3D rendering
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'spatial_understanding': 0.35,
            'rotation_angle_accuracy': 0.35,
            'view_consistency': 0.20,
            'rendering_quality': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate 3D rotation view."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # 1. 3D spatial understanding (35%): Check voxel structure match
        spatial_score = self._evaluate_spatial_understanding(gen_final, gt_final)
        scores['spatial_understanding'] = spatial_score
        
        # 2. Rotation angle accuracy (35%): Check 180° rotation
        rotation_score = self._evaluate_rotation_angle(first_frame, gen_final, gt_final)
        scores['rotation_angle_accuracy'] = rotation_score
        
        # 3. View consistency (20%): Same structure from opposite view
        consistency_score = self._evaluate_view_consistency(first_frame, gen_final)
        scores['view_consistency'] = consistency_score
        
        # 4. Rendering quality (10%): 3D rendering quality
        rendering_score = self._evaluate_rendering_quality(gen_final, gt_final)
        scores['rendering_quality'] = rendering_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_voxels(self, frame: np.ndarray) -> np.ndarray:
        """Detect blue voxel regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Light blue voxels
        lower_blue = np.array([90, 50, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return blue_mask
    
    def _evaluate_spatial_understanding(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate 3D structure understanding."""
        gen_voxels = self._detect_voxels(gen_frame)
        gt_voxels = self._detect_voxels(gt_frame)
        
        if np.sum(gt_voxels > 0) == 0:
            return 0.5
        
        # IoU of voxel regions
        intersection = np.sum((gen_voxels > 0) & (gt_voxels > 0))
        union = np.sum((gen_voxels > 0) | (gt_voxels > 0))
        
        if union > 0:
            return intersection / union
        return 0.5
    
    def _evaluate_rotation_angle(self, first_frame: np.ndarray, gen_final: np.ndarray, 
                                  gt_final: np.ndarray) -> float:
        """Evaluate if 180° rotation was achieved."""
        # Compare generated final with GT final
        gen_voxels = self._detect_voxels(gen_final)
        gt_voxels = self._detect_voxels(gt_final)
        
        if np.sum(gt_voxels > 0) == 0:
            return 0.5
        
        # IoU
        intersection = np.sum((gen_voxels > 0) & (gt_voxels > 0))
        union = np.sum((gen_voxels > 0) | (gt_voxels > 0))
        
        if union > 0:
            return intersection / union
        return 0.5
    
    def _evaluate_view_consistency(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if structure is consistent (same voxel count)."""
        first_voxels = self._detect_voxels(first_frame)
        final_voxels = self._detect_voxels(gen_final)
        
        first_area = np.sum(first_voxels > 0)
        final_area = np.sum(final_voxels > 0)
        
        if first_area == 0:
            return 0.5
        
        # Area should be similar (same structure, different view)
        ratio = final_area / first_area
        
        if 0.7 <= ratio <= 1.3:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.7
        else:
            return max(0.2, 1.0 - abs(1 - ratio))
    
    def _evaluate_rendering_quality(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate 3D rendering quality."""
        gen_voxels = self._detect_voxels(gen_frame)
        gt_voxels = self._detect_voxels(gt_frame)
        
        # Check if voxels have clear edges (black borders)
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        
        # Edges should be present around voxels
        dilated_voxels = cv2.dilate(gen_voxels, np.ones((5, 5), np.uint8))
        edge_near_voxels = np.sum((gen_edges > 0) & (dilated_voxels > 0))
        
        if np.sum(gen_voxels > 0) > 0:
            edge_ratio = edge_near_voxels / np.sum(gen_voxels > 0)
            return min(1.0, edge_ratio * 2)
        
        return 0.5
