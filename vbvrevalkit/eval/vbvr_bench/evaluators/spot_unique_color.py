"""Evaluator for G-138_spot_unique_non_repeated_color_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-138_spot_unique_non_repeated_color_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class SpotUniqueColorEvaluator(BaseEvaluator):
    """
    G-138: Spot unique non-repeated color evaluator.
    
    Rule-based evaluation:
    - Color uniqueness identification (50%): Find color appearing only once
    - Shape localization accuracy (30%): Accurate outline of unique shape
    - Visual annotation quality (15%): Outline complete and visible
    - Understanding accuracy (5%): Understand "unique" vs "repeated"
    """
    
    TASK_WEIGHTS = {
        'uniqueness': 0.50,
        'localization': 0.30,
        'annotation': 0.15,
        'understanding': 0.05
    }
    
    def _detect_colored_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored shapes and their colors."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        raw_shapes = []
        
        # Define color ranges with lower saturation threshold (50 instead of 100)
        # to catch semi-transparent shapes
        # Note: ranges are non-overlapping to avoid duplicate detection
        color_ranges = {
            'red': [([0, 50, 50], [5, 255, 255]), ([170, 50, 50], [180, 255, 255])],
            'orange': [([5, 50, 50], [15, 255, 255])],
            'yellow': [([15, 50, 50], [35, 255, 255])],
            'green': [([35, 50, 50], [85, 255, 255])],
            'cyan': [([85, 50, 50], [100, 255, 255])],
            'blue': [([100, 50, 50], [130, 255, 255])],
            'magenta': [([140, 50, 50], [170, 255, 255])],
        }
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                raw_shapes.append({
                    'color': color_name,
                    'center': (cx, cy),
                    'area': area
                })
        
        # Remove duplicates (shapes with very similar centers detected as different colors)
        shapes = []
        for s in raw_shapes:
            is_dup = False
            for existing in shapes:
                dist = np.sqrt((s['center'][0] - existing['center'][0])**2 + 
                              (s['center'][1] - existing['center'][1])**2)
                if dist < 50:  # Same shape detected twice
                    is_dup = True
                    break
            if not is_dup:
                shapes.append(s)
        
        return shapes
    
    def _detect_outline_marking(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black outline markings around shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect black pixels (outlines are black)
        black_mask = (gray < 30).astype(np.uint8) * 255
        
        # Use morphological operations to connect outline fragments
        kernel = np.ones((3, 3), np.uint8)
        black_mask = cv2.dilate(black_mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Black outlines can be thin (small area) or surround a shape (larger area)
            if area < 30:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # For outline markings, the center might be the center of the outlined shape
            # Get bounding rect to find the approximate center of the marked region
            x, y, w, h = cv2.boundingRect(cnt)
            centers.append((x + w // 2, y + h // 2))
        
        return centers
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate spot unique color task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect shapes and markings
        gen_shapes = self._detect_colored_shapes(last_frame)
        gt_shapes = self._detect_colored_shapes(gt_last)
        
        gen_markings = self._detect_outline_marking(last_frame)
        gt_markings = self._detect_outline_marking(gt_last)
        
        # 1. Color uniqueness: Find unique color and check if marked
        color_counts_gt = {}
        for shape in gt_shapes:
            color_counts_gt[shape['color']] = color_counts_gt.get(shape['color'], 0) + 1
        
        # Find unique colors (appearing once)
        unique_colors = [c for c, count in color_counts_gt.items() if count == 1]
        
        # Check if generated marking is near a unique-colored shape
        if gen_markings and gen_shapes:
            marked_unique = False
            for marking in gen_markings:
                for shape in gen_shapes:
                    dist = np.sqrt((marking[0] - shape['center'][0])**2 + 
                                  (marking[1] - shape['center'][1])**2)
                    # More lenient distance threshold
                    if dist < 100 and shape['color'] in unique_colors:
                        marked_unique = True
                        break
            scores['uniqueness'] = 1.0 if marked_unique else 0.5
        else:
            # Rule-based fallback: check if any marking exists near any shape
            scores['uniqueness'] = 0.3 if gen_markings else 0.0
        
        # 2. Localization: Compare marking positions with GT
        if gen_markings and gt_markings:
            matched = 0
            for gm in gen_markings:
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    # Very close match (GT vs GT case)
                    if dist < 15:
                        matched += 1
                        break
                    elif dist < 80:  # More lenient threshold
                        matched += 0.8
                        break
            scores['localization'] = min(1.0, matched / max(len(gt_markings), 1))
        else:
            # Rule-based: no GT markings means no unique color expected
            scores['localization'] = 0.5 if not gt_markings else 0.0
        
        # 3. Annotation quality: Check outline presence using IoU
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        black_mask_gen = (gray_gen < 50).astype(np.uint8)
        black_mask_gt = (gray_gt < 50).astype(np.uint8)
        
        black_overlap = np.sum((black_mask_gen > 0) & (black_mask_gt > 0))
        black_union = np.sum((black_mask_gen > 0) | (black_mask_gt > 0))
        
        scores['annotation'] = black_overlap / black_union if black_union > 0 else 0.5
        
        # 4. Understanding: Check if only unique shapes are marked
        if gen_markings and gen_shapes:
            correct_marks = 0
            total_marks = len(gen_markings)
            for marking in gen_markings:
                for shape in gen_shapes:
                    dist = np.sqrt((marking[0] - shape['center'][0])**2 + 
                                  (marking[1] - shape['center'][1])**2)
                    if dist < 100:  # More lenient threshold
                        color_counts_gen = {}
                        for s in gen_shapes:
                            color_counts_gen[s['color']] = color_counts_gen.get(s['color'], 0) + 1
                        if color_counts_gen.get(shape['color'], 0) == 1:
                            correct_marks += 1
                        break
            scores['understanding'] = correct_marks / total_marks if total_marks > 0 else 0.8
        else:
            # Rule-based fallback
            scores['understanding'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
