"""Evaluator for G-223_highlight_horizontal_lines_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-223_highlight_horizontal_lines_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class HighlightHorizontalLinesEvaluator(BaseEvaluator):
    """
    G-223: Highlight horizontal lines evaluator.
    
    Rule-based evaluation:
    - Horizontal line identification accuracy (40%): All horizontal lines found
    - Marking completeness (30%): All horizontal lines marked
    - Marking position accuracy (20%): Circles centered on line midpoints
    - Visual annotation quality (10%): Black circles proper
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'completeness': 0.30,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_horizontal_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect horizontal line segments using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find non-white regions (colored lines)
        non_white = (gray < 250).astype(np.uint8) * 255
        contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        horizontal_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            
            # Horizontal line: width >> height (aspect > 5)
            if aspect > 5:
                midpoint = (x + w // 2, y + h // 2)
                horizontal_lines.append({
                    'start': (x, y),
                    'end': (x + w, y),
                    'midpoint': midpoint,
                    'length': w
                })
        
        return horizontal_lines
    
    def _detect_black_markings(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black circle markings."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Black circles can be large (up to 50000 area for big marking circles)
            if area < 30:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
        
        return centers
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate highlight horizontal lines task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect markings
        gen_markings = self._detect_black_markings(last_frame)
        gt_markings = self._detect_black_markings(gt_last)
        
        # Detect horizontal lines in both frames
        gen_lines = self._detect_horizontal_lines(last_frame)
        gt_lines = self._detect_horizontal_lines(gt_last)
        
        # Count expected horizontal lines (lines with y1 = y2)
        expected_horizontal_count = len([l for l in gt_lines if abs(l['start'][1] - l['end'][1]) < 10])
        
        # 1. Identification: Check if markings are on horizontal lines (40%)
        # Rule: Must correctly identify horizontal lines (y1 = y2) vs vertical (x1 = x2)
        if gen_markings and gen_lines:
            on_line_count = 0
            for marking in gen_markings:
                for line in gen_lines:
                    dist = np.sqrt((marking[0] - line['midpoint'][0])**2 + 
                                  (marking[1] - line['midpoint'][1])**2)
                    if dist < 80:  # More lenient threshold
                        on_line_count += 1
                        break
            scores['identification'] = on_line_count / len(gen_markings) if gen_markings else 0.0
        else:
            # No markings - score based on whether horizontal lines exist
            scores['identification'] = 0.5 if expected_horizontal_count == 0 else 0.0
        
        # 2. Completeness: Compare marking counts (30%)
        # Rule: All horizontal lines must be marked (recall = 100%)
        if gt_markings:
            count_diff = abs(len(gen_markings) - len(gt_markings))
            # Exact match or very close gets full score
            if count_diff == 0:
                scores['completeness'] = 1.0
            elif count_diff == 1:
                scores['completeness'] = 0.7
            else:
                scores['completeness'] = max(0.3, 1.0 - count_diff * 0.2)
        else:
            # No GT markings means no horizontal lines expected
            scores['completeness'] = 1.0 if len(gen_markings) == 0 else 0.5
        
        # 3. Position accuracy: Compare marking positions with GT
        if gen_markings and gt_markings:
            matched = 0
            total_dist = 0
            for gm in gen_markings:
                min_dist = float('inf')
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    min_dist = min(min_dist, dist)
                # Very close match (< 10 pixels) counts as perfect match
                if min_dist < 10:
                    matched += 1
                    total_dist += 0
                elif min_dist < 80:  # More lenient threshold
                    matched += 1
                    total_dist += min_dist
            
            if matched > 0:
                avg_dist = total_dist / matched
                scores['position'] = max(0, 1.0 - avg_dist / 40.0)
            else:
                scores['position'] = 0.3
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Black pixel IoU
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        black_mask_gen = (gray_gen < 50).astype(np.uint8)
        black_mask_gt = (gray_gt < 50).astype(np.uint8)
        
        black_overlap = np.sum((black_mask_gen > 0) & (black_mask_gt > 0))
        black_union = np.sum((black_mask_gen > 0) | (black_mask_gt > 0))
        
        scores['annotation'] = black_overlap / black_union if black_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Mapping of task names to evaluators
OPEN60_EVALUATORS_PART3 = {
    'G-134_select_next_figure_large_small_alternating_sequence_data-generator': SelectNextFigureLargeSmallEvaluator,
    'G-138_spot_unique_non_repeated_color_data-generator': SpotUniqueColorEvaluator,
    'G-158_identify_all_hollow_points_data-generator': IdentifyAllHollowPointsEvaluator,
    'G-168_identify_nearest_to_square_rectangle_data-generator': IdentifyNearestSquareRectangleEvaluator,
    'G-169_locate_intersection_of_segments_data-generator': LocateSegmentIntersectionEvaluator,
    'G-189_draw_midpoint_perpendicular_line_data-generator': DrawMidpointPerpendicularEvaluator,
    'G-194_construct_concentric_ring_data-generator': ConstructConcentricRingEvaluator,
    'G-206_identify_pentagons_data-generator': IdentifyPentagonsEvaluator,
    'G-222_mark_tangent_point_of_circles_data-generator': MarkTangentPointEvaluator,
    'G-223_highlight_horizontal_lines_data-generator': HighlightHorizontalLinesEvaluator,
