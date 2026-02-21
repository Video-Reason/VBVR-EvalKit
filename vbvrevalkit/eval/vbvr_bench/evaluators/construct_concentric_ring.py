"""Evaluator for G-194_construct_concentric_ring_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-194_construct_concentric_ring_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class ConstructConcentricRingEvaluator(BaseEvaluator):
    """
    G-194: Construct concentric ring evaluator.
    
    Rule-based evaluation:
    - Center alignment accuracy (35%): Both circles centered at image center
    - Circle attribute fidelity (35%): Radius, color, shape preserved
    - Concentric structure correctness (20%): Proper concentric geometry
    - Animation smoothness (10%): Smooth movement
    """
    
    TASK_WEIGHTS = {
        'center': 0.35,
        'fidelity': 0.35,
        'structure': 0.20,
        'smoothness': 0.10
    }
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the frame - optimized for concentric ring detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # For concentric rings, we expect 2 circles with the same center
        # Use edge detection to find circle outlines
        edges = cv2.Canny(gray, 50, 150)
        
        # Use contour detection for more reliable circle finding
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Skip small contours
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 100:  # Skip small perimeters
                continue
            
            # Check circularity
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.6:  # Reasonably circular
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # Verify the enclosing circle fits well
                if radius > 50:  # Minimum radius for concentric rings
                    detected.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius)
                    })
        
        # If contour method fails, try Hough circles with stricter params
        if len(detected) < 2:
            # Use stricter parameters to avoid false positives
            for param2 in [50, 40, 30]:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                            param1=100, param2=param2, minRadius=50, maxRadius=500)
                if circles is not None and len(circles[0]) >= 2:
                    circles = np.uint16(np.around(circles))
                    detected = []
                    for i in circles[0, :]:
                        detected.append({
                            'center': (int(i[0]), int(i[1])),
                            'radius': int(i[2])
                        })
                    break
        
        # Remove duplicates (circles with very similar centers and radii)
        if len(detected) > 2:
            unique = []
            for d in detected:
                is_dup = False
                for u in unique:
                    center_dist = np.sqrt((d['center'][0] - u['center'][0])**2 + 
                                         (d['center'][1] - u['center'][1])**2)
                    radius_diff = abs(d['radius'] - u['radius'])
                    if center_dist < 30 and radius_diff < 30:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(d)
            detected = unique
        
        return detected
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate construct concentric ring task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect circles
        gen_circles = self._detect_circles(last_frame)
        gt_circles = self._detect_circles(gt_last)
        
        frame_center = (last_frame.shape[1] // 2, last_frame.shape[0] // 2)
        
        # 1. Center alignment: Check if circles are centered at image center (512, 512)
        if gen_circles:
            center_dists = []
            for c in gen_circles:
                dist = np.sqrt((c['center'][0] - frame_center[0])**2 + 
                              (c['center'][1] - frame_center[1])**2)
                center_dists.append(dist)
            avg_dist = np.mean(center_dists)
            # Rule: circles should be within 5 pixels of center for perfect score
            if avg_dist < 5:
                scores['center'] = 1.0
            elif avg_dist < 15:
                scores['center'] = 0.9
            else:
                scores['center'] = max(0, 1.0 - avg_dist / 100.0)
        else:
            # No circles detected - check if any circular content exists
            gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            scores['center'] = 0.3 if np.sum(edges > 0) > 1000 else 0.0
        
        # 2. Fidelity: Compare circle count and radii with GT
        if gen_circles and gt_circles:
            # Count match - should have 2 circles for concentric rings
            count_match = max(0, 1.0 - abs(len(gen_circles) - len(gt_circles)) / max(len(gt_circles), 1))
            
            # Radius comparison - radii should be preserved (rule: error <= 3 pixels)
            gen_radii = sorted([c['radius'] for c in gen_circles])
            gt_radii = sorted([c['radius'] for c in gt_circles])
            
            if len(gen_radii) == len(gt_radii):
                radius_diffs = [abs(g - gt) for g, gt in zip(gen_radii, gt_radii)]
                avg_radius_diff = np.mean(radius_diffs)
                if avg_radius_diff <= 3:
                    radius_match = 1.0
                elif avg_radius_diff <= 8:
                    radius_match = 0.8
                else:
                    radius_match = max(0, 1.0 - avg_radius_diff / 50.0)
            else:
                # Partial match based on common radii
                min_len = min(len(gen_radii), len(gt_radii))
                if min_len > 0:
                    radius_diffs = [abs(g - gt) for g, gt in zip(gen_radii[:min_len], gt_radii[:min_len])]
                    avg_radius_diff = np.mean(radius_diffs)
                    radius_match = max(0, 1.0 - avg_radius_diff / 50.0) * 0.7
                else:
                    radius_match = 0.5
            
            scores['fidelity'] = 0.5 * count_match + 0.5 * radius_match
        elif gen_circles:
            # Generated has circles but GT doesn't (shouldn't happen for concentric task)
            scores['fidelity'] = 0.3
        else:
            # No circles detected
            scores['fidelity'] = 0.0
        
        # 3. Concentric structure: Check if circles share same center (rule: centers must coincide)
        if len(gen_circles) >= 2:
            centers = [c['center'] for c in gen_circles]
            # Calculate max distance between any two circle centers
            max_center_dist = 0
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                  (centers[i][1] - centers[j][1])**2)
                    max_center_dist = max(max_center_dist, dist)
            
            # Rule: for perfect concentric, centers should coincide (error <= 5 pixels)
            if max_center_dist <= 5:
                scores['structure'] = 1.0
            elif max_center_dist <= 10:
                scores['structure'] = 0.9
            else:
                scores['structure'] = max(0, 1.0 - max_center_dist / 50.0)
        elif len(gen_circles) == 1:
            scores['structure'] = 0.2  # Single circle cannot form concentric structure
        else:
            scores['structure'] = 0.0
        
        # 4. Smoothness: Analyze motion through video (both circles should move simultaneously)
        if len(video_frames) >= 3:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            if motion_scores:
                variance = np.var(motion_scores)
                # Low variance = smooth, consistent motion
                scores['smoothness'] = max(0.5, 1.0 - variance / 1000.0)
            else:
                scores['smoothness'] = 0.8
        else:
            scores['smoothness'] = 0.8
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
