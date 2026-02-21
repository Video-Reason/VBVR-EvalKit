"""Evaluator for O-27_move_2_object_to_2_target_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-27_move_2_object_to_2_target_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class MoveObjectsToTargetEvaluator(BaseEvaluator):
    """
    O-27: Move 2 Objects to 2 Targets
    
    Task: Animate two colored balls (pink and blue) sliding to their 
    matching colored target rings simultaneously.
    
    Rule-based evaluation:
    1. Color matching (50%) - Pink to pink, blue to blue
    2. Path and motion (15%) - Straight line, smooth
    3. Movement synchronization (20%) - Start/end together
    4. Visual completeness (15%) - Balls and targets preserved
    """
    
    TASK_WEIGHTS = {
        'color_matching': 0.50,
        'path_motion': 0.15,
        'synchronization': 0.20,
        'completeness': 0.15
    }
    
    def _find_color_centers(self, frame: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """Find centers of pink and blue objects."""
        if len(frame.shape) != 3:
            return {'pink': None, 'blue': None}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        centers = {}
        
        # Pink detection
        lower_pink = np.array([140, 50, 80])
        upper_pink = np.array([170, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        if np.sum(pink_mask > 0) > 100:
            coords = np.where(pink_mask > 0)
            centers['pink'] = (float(np.mean(coords[1])), float(np.mean(coords[0])))
        else:
            centers['pink'] = None
        
        # Blue detection
        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        if np.sum(blue_mask > 0) > 100:
            coords = np.where(blue_mask > 0)
            centers['blue'] = (float(np.mean(coords[1])), float(np.mean(coords[0])))
        else:
            centers['blue'] = None
        
        return centers
    
    def _analyze_motion_smoothness(self, frames: List[np.ndarray]) -> float:
        """Analyze if motion is smooth and continuous."""
        if len(frames) < 3:
            return 0.5
        
        trajectories = {'pink': [], 'blue': []}
        
        for frame in frames:
            centers = self._find_color_centers(frame)
            for color in ['pink', 'blue']:
                if centers[color] is not None:
                    trajectories[color].append(centers[color])
        
        smoothness_scores = []
        for color in ['pink', 'blue']:
            points = trajectories[color]
            if len(points) < 3:
                continue
            
            disps = []
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                disps.append(np.sqrt(dx**2 + dy**2))
            
            if len(disps) < 2:
                continue
            
            mean_disp = np.mean(disps)
            std_disp = np.std(disps)
            
            if mean_disp < 0.5:
                smoothness_scores.append(0.4)
            else:
                smoothness_scores.append(max(0.3, 1 - (std_disp / mean_disp)))
        
        return float(np.mean(smoothness_scores)) if smoothness_scores else 0.5
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate dual object movement to targets."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Color matching: Compare final positions
        gen_centers = self._find_color_centers(gen_final)
        gt_centers = self._find_color_centers(gt_final)
        
        total_dist = 0
        count = 0
        for color in ['pink', 'blue']:
            if gen_centers[color] is not None and gt_centers[color] is not None:
                dx = gen_centers[color][0] - gt_centers[color][0]
                dy = gen_centers[color][1] - gt_centers[color][1]
                dist = np.sqrt(dx**2 + dy**2)
                total_dist += dist
                count += 1
        
        if count > 0:
            avg_dist = total_dist / count
            scores['color_matching'] = max(0, 1.0 - avg_dist / 50.0)
        else:
            scores['color_matching'] = 0.3
        
        # 2. Path motion: Analyze smoothness
        scores['path_motion'] = self._analyze_motion_smoothness(video_frames)
        
        # 3. Synchronization: Check if both objects move together
        if len(video_frames) >= 3:
            first_centers = self._find_color_centers(video_frames[0])
            mid_centers = self._find_color_centers(video_frames[len(video_frames)//2])
            
            pink_moved = False
            blue_moved = False
            
            if first_centers['pink'] is not None and mid_centers['pink'] is not None:
                pink_dist = np.sqrt((mid_centers['pink'][0] - first_centers['pink'][0])**2 + 
                                   (mid_centers['pink'][1] - first_centers['pink'][1])**2)
                pink_moved = pink_dist > 10
            
            if first_centers['blue'] is not None and mid_centers['blue'] is not None:
                blue_dist = np.sqrt((mid_centers['blue'][0] - first_centers['blue'][0])**2 + 
                                   (mid_centers['blue'][1] - first_centers['blue'][1])**2)
                blue_moved = blue_dist > 10
            
            # Both should move together
            if pink_moved and blue_moved:
                scores['synchronization'] = 1.0
            elif pink_moved or blue_moved:
                scores['synchronization'] = 0.2  # Detection failed
            else:
                scores['synchronization'] = 0.3
        else:
            scores['synchronization'] = 0.2  # Detection failed
        
        # 4. Completeness: Check objects are preserved
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['completeness'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['completeness'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
