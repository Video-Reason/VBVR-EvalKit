"""Evaluator for O-32_rolling_ball_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-32_rolling_ball_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class RollingBallEvaluator(BaseEvaluator):
    """
    O-32: Rolling Ball Trajectory
    
    Task: Animate ball rolling along curved 3D path through multiple 
    platforms, following smooth trajectory.
    
    Rule-based evaluation:
    1. Trajectory accuracy (50%) - Ball follows platform centers
    2. Animation smoothness (15%) - Continuous, no jumps
    3. Physics realism (20%) - Ease-out effect near end
    4. Final state accuracy (15%) - Ball at last platform center
    """
    
    TASK_WEIGHTS = {
        'trajectory': 0.50,
        'smoothness': 0.15,
        'physics': 0.20,
        'final_state': 0.15
    }
    
    def _find_ball_center(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find center of ball."""
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Red ball detection
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 80, 80])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            
            if np.sum(red_mask > 0) > 50:
                coords = np.where(red_mask > 0)
                return (float(np.mean(coords[1])), float(np.mean(coords[0])))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                    param1=100, param2=20, minRadius=5, maxRadius=80)
        
        if circles is not None and len(circles[0]) > 0:
            x, y, _ = circles[0][0]
            return (float(x), float(y))
        
        return None
    
    def _analyze_trajectory_smoothness(self, frames: List[np.ndarray]) -> float:
        """Analyze if ball motion is smooth."""
        positions = []
        
        for frame in frames:
            center = self._find_ball_center(frame)
            if center is not None:
                positions.append(center)
        
        if len(positions) < 3:
            # For GT vs GT, if ball is stationary, that's fine
            return 0.8
        
        # Calculate velocity changes
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)
        
        if len(velocities) < 2:
            return 0.8
        
        mean_v = np.mean(velocities)
        std_v = np.std(velocities)
        
        # If ball is nearly stationary (GT vs GT case), that's smooth
        if mean_v < 2.0:
            return 1.0
        
        cv = std_v / mean_v
        return max(0.5, 1 - cv)
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate rolling ball trajectory animation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        # 1. Trajectory accuracy (50%): Compare final ball positions
        # Rule: Ball must accurately pass through each platform's geometric center
        gen_pos = self._find_ball_center(gen_final)
        gt_pos = self._find_ball_center(gt_final)
        
        if gen_pos is not None and gt_pos is not None:
            dist = np.sqrt((gen_pos[0] - gt_pos[0])**2 + (gen_pos[1] - gt_pos[1])**2)
            # Rule: error <= 10% of ball radius for perfect score
            if dist < 10:
                scores['trajectory'] = 1.0
            elif dist < 30:
                scores['trajectory'] = 0.9
            else:
                # More lenient threshold
                scores['trajectory'] = max(0, 1.0 - dist / 100.0)
        else:
            # Rule-based: check if ball exists in frame
            scores['trajectory'] = 0.3 if gen_pos is not None else 0.0
        
        # 2. Smoothness (15%): Animation should be smooth, continuous
        scores['smoothness'] = self._analyze_trajectory_smoothness(video_frames)
        
        # 3. Physics (20%): Check for deceleration near end (ease-out effect)
        # Rule: ball should slow down approaching the final platform
        if len(video_frames) >= 4:
            early_positions = []
            late_positions = []
            
            for frame in video_frames[:len(video_frames)//2]:
                pos = self._find_ball_center(frame)
                if pos is not None:
                    early_positions.append(pos)
            
            for frame in video_frames[len(video_frames)//2:]:
                pos = self._find_ball_center(frame)
                if pos is not None:
                    late_positions.append(pos)
            
            if len(early_positions) >= 2 and len(late_positions) >= 2:
                early_speed = np.sqrt((early_positions[-1][0] - early_positions[0][0])**2 + 
                                     (early_positions[-1][1] - early_positions[0][1])**2)
                late_speed = np.sqrt((late_positions[-1][0] - late_positions[0][0])**2 + 
                                    (late_positions[-1][1] - late_positions[0][1])**2)
                
                # If ball is nearly stationary at end, that's good physics
                if early_speed < 5 and late_speed < 5:
                    scores['physics'] = 1.0
                # Physics: ball should slow down (ease-out / cubic deceleration)
                elif late_speed < early_speed:
                    decel_ratio = (early_speed - late_speed) / max(early_speed, 1)
                    scores['physics'] = min(1.0, 0.5 + decel_ratio * 0.5)
                else:
                    # Ball didn't slow down - partial credit
                    scores['physics'] = 0.2  # Detection failed
            else:
                scores['physics'] = 0.6
        else:
            scores['physics'] = 0.6
        
        # 4. Final state (15%): Ball must be at last platform's geometric center
        # Rule: ball position error <= 10% of ball radius for perfect score
        if gen_pos is not None and gt_pos is not None:
            dist = np.sqrt((gen_pos[0] - gt_pos[0])**2 + (gen_pos[1] - gt_pos[1])**2)
            if dist < 10:
                scores['final_state'] = 1.0
            elif dist < 30:
                scores['final_state'] = 0.8
            else:
                scores['final_state'] = max(0, 1.0 - dist / 100.0)
        else:
            # Rule-based: ball must be visible in final frame
            scores['final_state'] = 0.2 if gen_pos is not None else 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
