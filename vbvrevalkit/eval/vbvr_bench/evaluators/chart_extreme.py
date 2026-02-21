"""Evaluator for G-29_chart_extreme_with_data_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-29_chart_extreme_with_data_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class ChartExtremeEvaluator(BaseEvaluator):
    """
    G-29: Chart extreme with data evaluator.
    
    Evaluates:
    - Extreme identification accuracy (40%): Correct max/min identified
    - Marking correctness (35%): Red rectangle accurately marks target
    - Visual normality (15%): Border complete and accurate
    - Chart fidelity (10%): Chart elements preserved
    """
    
    TASK_WEIGHTS = {
        'extreme_id': 0.40,
        'marking': 0.35,
        'visual': 0.15,
        'fidelity': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate chart extreme task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect red rectangles in both frames
        gen_red_regions = self._detect_red_rectangle(last_frame)
        gt_red_regions = self._detect_red_rectangle(gt_last)
        
        # 1. Extreme identification (40%): Check if red rectangle is in correct location
        extreme_score = self._evaluate_extreme_identification(gen_red_regions, gt_red_regions, last_frame, gt_last)
        scores['extreme_id'] = extreme_score
        
        # 2. Marking correctness (35%): Red rectangle position and completeness
        marking_score = self._evaluate_marking(gen_red_regions, gt_red_regions)
        scores['marking'] = marking_score
        
        # 3. Visual normality (15%): Border completeness
        visual_score = self._evaluate_visual_normality(last_frame, gen_red_regions)
        scores['visual'] = visual_score
        
        # 4. Chart fidelity (10%): Chart elements preserved
        fidelity_score = self._evaluate_chart_fidelity(last_frame, gt_last, gen_red_regions, gt_red_regions)
        scores['fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_rectangle(self, frame: np.ndarray) -> List[Dict]:
        """Detect red rectangular borders in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range (two ranges due to hue wrap-around)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter noise
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's rectangular (not filled)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            # Get center
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            regions.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area,
                'contour': contour,
                'is_border': fill_ratio < 0.5  # Border has low fill ratio
            })
        
        return regions
    
    def _evaluate_extreme_identification(self, gen_regions: List[Dict], gt_regions: List[Dict], 
                                         gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Check if the correct extreme element is marked."""
        if not gen_regions or not gt_regions:
            return 0.0  # STRICT: Must have marking in both
        
        # Find the main red marking in both
        gen_main = max(gen_regions, key=lambda r: r['area'])
        gt_main = max(gt_regions, key=lambda r: r['area'])
        
        # Compare centers
        gen_center = gen_main['center']
        gt_center = gt_main['center']
        
        # Calculate distance normalized by frame size
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        # STRICTER: Score based on distance
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.7
        elif normalized_dist < 0.20:
            return 0.3
        else:
            return 0.0  # STRICT: Wrong position
    
    def _evaluate_marking(self, gen_regions: List[Dict], gt_regions: List[Dict]) -> float:
        """Evaluate red rectangle marking accuracy."""
        if not gen_regions:
            return 0.0
        if not gt_regions:
            return 0.0  # STRICT: No GT to compare
        
        gen_main = max(gen_regions, key=lambda r: r['area'])
        gt_main = max(gt_regions, key=lambda r: r['area'])
        
        # Compare bounding boxes using IoU
        gen_bbox = gen_main['bbox']
        gt_bbox = gt_main['bbox']
        
        # Calculate IoU
        x1 = max(gen_bbox[0], gt_bbox[0])
        y1 = max(gen_bbox[1], gt_bbox[1])
        x2 = min(gen_bbox[0] + gen_bbox[2], gt_bbox[0] + gt_bbox[2])
        y2 = min(gen_bbox[1] + gen_bbox[3], gt_bbox[1] + gt_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        gen_area = gen_bbox[2] * gen_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]
        union = gen_area + gt_area - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _evaluate_visual_normality(self, frame: np.ndarray, red_regions: List[Dict]) -> float:
        """Evaluate if the red border is complete and proper."""
        if not red_regions:
            return 0.0
        
        main_region = max(red_regions, key=lambda r: r['area'])
        
        # Check if it forms a complete rectangle (4 sides)
        contour = main_region['contour']
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # A rectangle should have 4 vertices
        if len(approx) == 4:
            return 1.0
        elif len(approx) >= 3 and len(approx) <= 6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_chart_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                                  gen_regions: List[Dict], gt_regions: List[Dict]) -> float:
        """Evaluate if chart elements are preserved."""
        # Mask out red regions and compare the rest
        gen_mask = np.ones(gen_frame.shape[:2], dtype=np.uint8) * 255
        gt_mask = np.ones(gt_frame.shape[:2], dtype=np.uint8) * 255
        
        for region in gen_regions:
            cv2.drawContours(gen_mask, [region['contour']], -1, 0, -1)
        for region in gt_regions:
            cv2.drawContours(gt_mask, [region['contour']], -1, 0, -1)
        
        # Compare non-red regions
        combined_mask = gen_mask & gt_mask
        
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        if np.sum(combined_mask > 0) == 0:
            return 0.0  # STRICT: No overlap to compare
        
        diff = np.abs(gen_gray.astype(float) - gt_gray.astype(float))
        masked_diff = diff[combined_mask > 0]
        
        avg_diff = np.mean(masked_diff) / 255.0
        return max(0, 1.0 - avg_diff * 2)
