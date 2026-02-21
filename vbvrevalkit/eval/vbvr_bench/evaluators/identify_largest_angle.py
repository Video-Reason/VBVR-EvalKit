"""Evaluator for G-218_identify_largest_angle_in_triangle_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-218_identify_largest_angle_in_triangle_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class IdentifyLargestAngleEvaluator(BaseEvaluator):
    """
    G-218: Identify largest angle in triangle evaluator.
    
    Rule-based evaluation:
    - Angle recognition correctness (40%): Identify largest angle vertex
    - Marking position precision (35%): Circle at correct vertex
    - Marking specification compliance (15%): Red circle, ~40px radius
    - Triangle preservation (10%): Original triangle unchanged
    """
    
    TASK_WEIGHTS = {
        'angle_recognition': 0.40,
        'marking_position': 0.35,
        'marking_specification': 0.15,
        'triangle_preservation': 0.10
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
        
        # 1. Angle recognition (40%)
        scores['angle_recognition'] = self._evaluate_angle_recognition(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking position (35%)
        scores['marking_position'] = self._evaluate_marking_position(
            first_frame, final_frame, gt_final_frame
        )
        
        # 3. Marking specification (15%)
        scores['marking_specification'] = self._evaluate_marking_spec(final_frame)
        
        # 4. Triangle preservation (10%)
        scores['triangle_preservation'] = self._evaluate_triangle_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_angle_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the largest angle vertex is identified."""
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + (circle[1] - gt_circle[1])**2)
                if dist < 30:
                    return 1.0
                elif dist < 60:
                    return 0.8
                elif dist < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Find largest angle vertex from triangle
        largest_vertex = self._find_largest_angle_vertex(first_frame)
        
        if largest_vertex is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - largest_vertex[0])**2 + (circle[1] - largest_vertex[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_position(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Evaluate circle position at vertex."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + (circle[1] - gt_circle[1])**2)
                return max(0.0, 1.0 - dist / 60)
        
        # Fallback: compare with detected largest vertex
        largest_vertex = self._find_largest_angle_vertex(first_frame)
        
        if largest_vertex is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - largest_vertex[0])**2 + (circle[1] - largest_vertex[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_spec(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check marking specification (~40px radius)."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 30 < r < 55:
            return 1.0
        elif 20 < r < 70:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_triangle_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if triangle is preserved."""
        # Detect triangle vertices
        first_vertices = self._detect_triangle_vertices(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_vertices = self._detect_triangle_vertices(final_no_red)
        
        if len(first_vertices) != 3:
            return 0.0
        
        if len(final_vertices) == 3:
            return 1.0
        elif len(final_vertices) >= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_largest_angle_vertex(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the vertex with the largest angle."""
        vertices = self._detect_triangle_vertices(frame)
        
        if len(vertices) != 3:
            return None
        
        # Calculate angles at each vertex
        angles = []
        for i in range(3):
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i+1) % 3])
            p3 = np.array(vertices[(i+2) % 3])
            
            v1 = p2 - p1
            v2 = p3 - p1
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append((angle, vertices[i]))
        
        # Return vertex with largest angle
        largest = max(angles, key=lambda x: x[0])
        return largest[1]
    
    def _detect_triangle_vertices(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect triangle vertices using corner detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find non-white regions (triangle lines)
        non_white = (gray < 250).astype(np.uint8) * 255
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
        
        # Get the largest contour (triangle outline)
        triangle = max(contours, key=cv2.contourArea)
        
        # First try polygon approximation (works for filled triangles)
        peri = cv2.arcLength(triangle, True)
        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(triangle, eps_factor * peri, True)
            if len(approx) == 3:
                return [tuple(pt[0]) for pt in approx]
        
        # If approximation fails (line-drawn triangles), use corner detection
        # Create a mask with the triangle contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [triangle], -1, 255, 3)
        
        # Detect corners using goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(mask, 20, 0.01, 30)
        
        if corners is not None and len(corners) >= 3:
            corner_pts = [(int(c[0][0]), int(c[0][1])) for c in corners]
            
            # Cluster nearby corners (corners along edges are close together)
            def cluster_corners(points, min_dist=50):
                """Cluster nearby points and return cluster centers."""
                if len(points) == 0:
                    return []
                
                clusters = []
                used = [False] * len(points)
                
                for i, p1 in enumerate(points):
                    if used[i]:
                        continue
                    
                    cluster = [p1]
                    used[i] = True
                    
                    for j, p2 in enumerate(points):
                        if not used[j]:
                            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            if dist < min_dist:
                                cluster.append(p2)
                                used[j] = True
                    
                    # Cluster center
                    cx = int(np.mean([p[0] for p in cluster]))
                    cy = int(np.mean([p[1] for p in cluster]))
                    clusters.append((cx, cy))
                
                return clusters
            
            clustered = cluster_corners(corner_pts, min_dist=60)
            
            if len(clustered) >= 3:
                # Find the 3 most extreme points (vertices of triangle)
                # Use convex hull to get the outer vertices
                pts_array = np.array(clustered, dtype=np.float32).reshape(-1, 1, 2)
                hull = cv2.convexHull(pts_array)
                
                if len(hull) >= 3:
                    # Sort by angle from centroid to get consistent ordering
                    hull_pts = [tuple(pt[0].astype(int)) for pt in hull]
                    return hull_pts[:3]
                
                return clustered[:3]
        
        return []
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
