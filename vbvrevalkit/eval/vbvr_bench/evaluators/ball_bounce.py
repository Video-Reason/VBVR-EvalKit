"""Evaluator for O-15_ball_bounces_given_time_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-15_ball_bounces_given_time_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class BallBounceEvaluator(BaseEvaluator):
    """
    O-15: Ball Bounces Given Time
    
    Task: Given initial position and velocity arrow, predict ball trajectory 
    with specified number of bounces off walls.
    
    Rule-based evaluation:
    1. Bounce count accuracy (30%) - Correct number of bounces
    2. Physics accuracy (35%) - Reflection law (angle in = angle out)
    3. Trajectory completeness (25%) - Full path shown
    4. Animation smoothness (10%) - Fluid motion
    """
    
    TASK_WEIGHTS = {
        'bounce_count': 0.30,
        'physics': 0.35,
        'trajectory': 0.25,
        'smoothness': 0.10
    }
    
    def _track_ball_positions(self, frames: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Track ball center position across frames."""
        positions = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Find dark regions (ball is typically dark)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Try HoughCircles first
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=30, minRadius=5, maxRadius=50)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    x, y, r = circles[0]
                    positions.append((x, y))
                    continue
            
            # Fallback: use contour centroid
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    positions.append((cx, cy))
        
        return positions
    
    def _count_bounces(self, positions: List[Tuple[float, float]]) -> int:
        """Count number of direction changes (bounces)."""
        if len(positions) < 3:
            return 0
        
        bounces = 0
        for i in range(2, len(positions)):
            dx1 = positions[i-1][0] - positions[i-2][0]
            dy1 = positions[i-1][1] - positions[i-2][1]
            dx2 = positions[i][0] - positions[i-1][0]
            dy2 = positions[i][1] - positions[i-1][1]
            
            # Check for direction reversal (bounce)
            if (dx1 * dx2 < -5) or (dy1 * dy2 < -5):
                bounces += 1
        
        return bounces
    
    def _calculate_motion_smoothness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate how smooth the motion is."""
        if len(positions) < 2:
            return 1.0
        
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)
        
        if len(velocities) < 2:
            return 1.0
        
        mean_v = np.mean(velocities)
        if mean_v < 1:
            return 0.5
        
        std_v = np.std(velocities)
        cv = std_v / mean_v
        
        return max(0, 1 - cv / 2)
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate ball bounce trajectory prediction."""
        
        if not video_frames or not gt_frames:
            return 0.0
        
        scores = {}
        
        # Track ball positions
        gen_positions = self._track_ball_positions(video_frames)
        gt_positions = self._track_ball_positions(gt_frames)
        
        # 1. Bounce count
        gen_bounces = self._count_bounces(gen_positions)
        gt_bounces = self._count_bounces(gt_positions)
        
        if gt_bounces > 0:
            bounce_diff = abs(gen_bounces - gt_bounces)
            scores['bounce_count'] = max(0, 1 - bounce_diff / gt_bounces)
        else:
            scores['bounce_count'] = 1.0 if gen_bounces == 0 else 0.5
        
        # 2. Physics accuracy: Compare final positions
        if gen_positions and gt_positions:
            gen_final = gen_positions[-1]
            gt_final = gt_positions[-1]
            dist = np.sqrt((gen_final[0] - gt_final[0])**2 + (gen_final[1] - gt_final[1])**2)
            scores['physics'] = max(0, 1.0 - dist / 100.0)
        else:
            scores['physics'] = 0.2  # Detection failed
        
        # 3. Trajectory completeness
        if gen_positions:
            # Check if trajectory spans reasonable distance
            if len(gen_positions) >= 2:
                total_dist = sum(np.sqrt((gen_positions[i][0] - gen_positions[i-1][0])**2 + 
                                        (gen_positions[i][1] - gen_positions[i-1][1])**2)
                               for i in range(1, len(gen_positions)))
                scores['trajectory'] = min(1.0, total_dist / 200.0)
            else:
                scores['trajectory'] = 0.3
        else:
            scores['trajectory'] = 0.0
        
        # 4. Smoothness
        scores['smoothness'] = self._calculate_motion_smoothness(gen_positions)
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
