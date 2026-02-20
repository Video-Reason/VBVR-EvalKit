"""
Specific evaluators for Open_60 tasks (Part 3).
These evaluators implement rule-based scoring for the third batch of Open_60 tasks.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class SelectNextFigureLargeSmallEvaluator(BaseEvaluator):
    """
    G-134: Select next figure large-small alternating evaluator.
    
    Rule-based evaluation:
    - Alternating pattern recognition (40%): Recognize big-small-big pattern
    - Shape type matching (30%): Correct shape type selected (same as sequence)
    - Size judgment (20%): Correct size (small) selected based on pattern
    - Visual annotation quality (10%): Red circle properly marks the selection
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.40,
        'shape_type': 0.30,
        'size': 0.20,
        'annotation': 0.10
    }
    
    def _detect_shapes_with_size(self, frame: np.ndarray) -> List[Dict]:
        """Detect shapes and their sizes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find colored areas (non-white, non-black)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Determine shape type
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)
            
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'square'
            elif vertices == 5:
                shape_type = 'pentagon'
            else:
                shape_type = 'circle'
            
            shapes.append({
                'type': shape_type,
                'center': (cx, cy),
                'area': area,
                'vertices': vertices
            })
        
        return shapes
    
    def _detect_red_circle_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # Reasonably circular
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _detect_marking_by_diff(self, first_frame: np.ndarray, final_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect marking by comparing first and final frames (for cases where shapes are red)."""
        # Compute difference
        diff = cv2.absdiff(first_frame, final_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours in difference
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Marking should be reasonably large
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate select next figure large-small alternating task."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_first = gt_first_frame
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect shapes and red marking
        gen_shapes = self._detect_shapes_with_size(last_frame)
        gt_shapes = self._detect_shapes_with_size(gt_last)
        
        # Try standard red marking detection first
        gen_marking = self._detect_red_circle_marking(last_frame)
        gt_marking = self._detect_red_circle_marking(gt_last)
        
        # If shapes are red, use frame difference to detect marking
        if first_frame is not None and gt_first is not None:
            gen_marking_diff = self._detect_marking_by_diff(first_frame, last_frame)
            gt_marking_diff = self._detect_marking_by_diff(gt_first, gt_last)
            
            # Use diff-based marking if available (more reliable when shapes are red)
            if gen_marking_diff is not None:
                gen_marking = gen_marking_diff
            if gt_marking_diff is not None:
                gt_marking = gt_marking_diff
        
        # 1. Pattern recognition: Check if marking is at correct position
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['pattern'] = max(0, 1.0 - dist / 100.0)
        elif gen_marking is not None:
            scores['pattern'] = 0.2  # Detection failed
        else:
            scores['pattern'] = 0.0
        
        # 2. Shape type matching: Check if marked shape has correct type
        if gen_shapes and gt_shapes:
            # Find shape nearest to marking
            gen_marked_shape = None
            if gen_marking is not None:
                min_dist = float('inf')
                for shape in gen_shapes:
                    dist = np.sqrt((shape['center'][0] - gen_marking[0])**2 + 
                                  (shape['center'][1] - gen_marking[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        gen_marked_shape = shape
            
            gt_marked_shape = None
            if gt_marking is not None:
                min_dist = float('inf')
                for shape in gt_shapes:
                    dist = np.sqrt((shape['center'][0] - gt_marking[0])**2 + 
                                  (shape['center'][1] - gt_marking[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        gt_marked_shape = shape
            
            if gen_marked_shape is not None and gt_marked_shape is not None:
                if gen_marked_shape['type'] == gt_marked_shape['type']:
                    scores['shape_type'] = 1.0
                else:
                    scores['shape_type'] = 0.3
            else:
                scores['shape_type'] = 0.2  # Detection failed
        else:
            scores['shape_type'] = 0.2  # Detection failed
        
        # 3. Size judgment: Check if marked shape has correct size category
        if gen_shapes and gt_shapes and gen_marking is not None and gt_marking is not None:
            # Get size of marked shapes
            gen_marked_area = 0
            for shape in gen_shapes:
                dist = np.sqrt((shape['center'][0] - gen_marking[0])**2 + 
                              (shape['center'][1] - gen_marking[1])**2)
                if dist < 100:
                    gen_marked_area = shape['area']
                    break
            
            gt_marked_area = 0
            for shape in gt_shapes:
                dist = np.sqrt((shape['center'][0] - gt_marking[0])**2 + 
                              (shape['center'][1] - gt_marking[1])**2)
                if dist < 100:
                    gt_marked_area = shape['area']
                    break
            
            if gen_marked_area > 0 and gt_marked_area > 0:
                # Compare relative sizes
                area_ratio = min(gen_marked_area, gt_marked_area) / max(gen_marked_area, gt_marked_area)
                scores['size'] = area_ratio
            else:
                scores['size'] = 0.2  # Detection failed
        else:
            scores['size'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Check red circle presence and quality
        if gen_marking is not None:
            hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
            hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
            red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
            
            red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
            red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
            
            scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        else:
            scores['annotation'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


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


class IdentifyAllHollowPointsEvaluator(BaseEvaluator):
    """
    G-158: Identify all hollow points evaluator.
    
    Rule-based evaluation:
    - Hollow point identification accuracy (40%): Distinguish hollow from solid
    - Marking completeness (30%): All hollow points marked
    - Marking position accuracy (20%): Circles centered on hollow points
    - Visual annotation quality (10%): Red circles proper
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'completeness': 0.30,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_hollow_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect hollow (unfilled) circular points."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find circles using Hough transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30, minRadius=5, maxRadius=50)
        
        hollow_points = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # Check if hollow (center is similar to background)
                # Sample center and edge
                if cy >= r and cy + r < frame.shape[0] and cx >= r and cx + r < frame.shape[1]:
                    center_val = gray[cy, cx]
                    edge_vals = []
                    for angle in range(0, 360, 45):
                        ex = int(cx + r * np.cos(np.radians(angle)))
                        ey = int(cy + r * np.sin(np.radians(angle)))
                        if 0 <= ex < frame.shape[1] and 0 <= ey < frame.shape[0]:
                            edge_vals.append(gray[ey, ex])
                    
                    if edge_vals:
                        edge_avg = np.mean(edge_vals)
                        # Hollow if center is brighter than edge (outline only)
                        if center_val > edge_avg + 30:
                            hollow_points.append((cx, cy))
        
        return hollow_points
    
    def _detect_red_markings(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect red circle markings."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
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
        """Evaluate identify all hollow points task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect red markings
        gen_markings = self._detect_red_markings(last_frame)
        gt_markings = self._detect_red_markings(gt_last)
        
        # 1. Identification: Compare number of markings
        if gt_markings:
            count_diff = abs(len(gen_markings) - len(gt_markings))
            scores['identification'] = max(0, 1.0 - count_diff * 0.2)
        else:
            scores['identification'] = 0.2  # Detection failed
        
        # 2. Completeness: Recall - how many GT markings are matched
        if gen_markings and gt_markings:
            matched = 0
            for gtm in gt_markings:
                for gm in gen_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    if dist < 40:
                        matched += 1
                        break
            scores['completeness'] = matched / len(gt_markings)
        else:
            scores['completeness'] = 0.5 if not gt_markings else 0.0
        
        # 3. Position accuracy: Average distance between matched markings
        if gen_markings and gt_markings:
            total_dist = 0
            matched_count = 0
            for gm in gen_markings:
                min_dist = float('inf')
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    total_dist += min_dist
                    matched_count += 1
            
            if matched_count > 0:
                avg_dist = total_dist / matched_count
                scores['position'] = max(0, 1.0 - avg_dist / 50.0)
            else:
                scores['position'] = 0.2  # Detection failed
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
        
        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
        
        scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class IdentifyNearestSquareRectangleEvaluator(BaseEvaluator):
    """
    G-168: Identify nearest to square rectangle evaluator.
    
    Rule-based evaluation:
    - Aspect ratio judgment accuracy (50%): Correct rectangle selected
    - Marking uniqueness (20%): Only one rectangle marked
    - Marking position accuracy (20%): Circle accurately surrounds rectangle
    - Visual annotation quality (10%): Red circle proper
    """
    
    TASK_WEIGHTS = {
        'aspect_ratio': 0.50,
        'uniqueness': 0.20,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_rectangles(self, frame: np.ndarray) -> List[Dict]:
        """Detect rectangles and calculate their aspect ratios."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            
            if len(approx) == 4:  # Rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = min(w, h) / max(w, h)  # 1.0 = perfect square
                
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                rectangles.append({
                    'center': (cx, cy),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'bounds': (x, y, w, h)
                })
        
        return rectangles
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate identify nearest to square rectangle task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect rectangles and markings
        gen_rects = self._detect_rectangles(last_frame)
        gt_rects = self._detect_rectangles(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Aspect ratio judgment: Check if marking is near the most square rectangle
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['aspect_ratio'] = max(0, 1.0 - dist / 100.0)
        elif gen_marking is not None and gen_rects:
            # Check if marked rectangle has highest aspect ratio
            marked_rect = None
            for rect in gen_rects:
                dist = np.sqrt((gen_marking[0] - rect['center'][0])**2 + 
                              (gen_marking[1] - rect['center'][1])**2)
                if dist < 100:
                    marked_rect = rect
                    break
            
            if marked_rect is not None:
                # Check if this is the most square one
                max_ratio = max(r['aspect_ratio'] for r in gen_rects)
                if marked_rect['aspect_ratio'] >= max_ratio - 0.1:
                    scores['aspect_ratio'] = 0.8
                else:
                    scores['aspect_ratio'] = 0.3
            else:
                scores['aspect_ratio'] = 0.3
        else:
            scores['aspect_ratio'] = 0.2  # Detection failed
        
        # 2. Uniqueness: Only one marking
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 100]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        # 3. Position accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Red pixel presence
        scores['annotation'] = min(1.0, np.sum(red_mask_gen > 0) / 500.0)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class LocateSegmentIntersectionEvaluator(BaseEvaluator):
    """
    G-169: Locate intersection of segments evaluator.
    
    Rule-based evaluation:
    - Intersection calculation accuracy (60%): Precise intersection point
    - Marking position accuracy (25%): Circle centered on intersection
    - Visual annotation quality (10%): Red circle proper
    - Marking uniqueness (5%): Only one point marked
    """
    
    TASK_WEIGHTS = {
        'calculation': 0.60,
        'position': 0.25,
        'annotation': 0.10,
        'uniqueness': 0.05
    }
    
    def _detect_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect line segments in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return (px, py)
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate locate segment intersection task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)  # Tight tolerance
        else:
            scores['calculation'] = 0.5 if gen_marking is None and gt_marking is None else 0.0
        
        # 2. Position accuracy: Same as calculation but with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 3. Annotation quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
        
        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
        
        scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Uniqueness: Only one marking
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 50]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class DrawMidpointPerpendicularEvaluator(BaseEvaluator):
    """
    G-189: Draw midpoint perpendicular line evaluator.
    
    Rule-based evaluation:
    - Midpoint identification accuracy (40%): Correct midpoint found
    - Perpendicular line position accuracy (30%): Line at x=width/2
    - Perpendicular line length/range (20%): Line spans between parallel lines
    - Visual quality (10%): Red line proper
    """
    
    TASK_WEIGHTS = {
        'midpoint': 0.40,
        'position': 0.30,
        'range': 0.20,
        'visual': 0.10
    }
    
    def _detect_red_line(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red vertical line."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find bounding box of red pixels
        points = np.where(red_mask > 0)
        if len(points[0]) < 10:
            return None
        
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        
        # Check if vertical (height > width)
        height = y_max - y_min
        width = x_max - x_min
        
        x_center = (x_min + x_max) // 2
        
        return {
            'x_center': x_center,
            'y_min': y_min,
            'y_max': y_max,
            'length': height,
            'is_vertical': height > width * 2
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate draw midpoint perpendicular task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect red lines
        gen_line = self._detect_red_line(last_frame)
        gt_line = self._detect_red_line(gt_last)
        
        # 1. Midpoint accuracy: Compare x-position
        if gen_line is not None and gt_line is not None:
            x_diff = abs(gen_line['x_center'] - gt_line['x_center'])
            scores['midpoint'] = max(0, 1.0 - x_diff / 30.0)
        elif gen_line is not None:
            # Check if at image center
            frame_center = last_frame.shape[1] // 2
            x_diff = abs(gen_line['x_center'] - frame_center)
            scores['midpoint'] = max(0, 1.0 - x_diff / 50.0)
        else:
            scores['midpoint'] = 0.0
        
        # 2. Position accuracy: Line should be vertical
        if gen_line is not None:
            scores['position'] = 1.0 if gen_line['is_vertical'] else 0.5
        else:
            scores['position'] = 0.0
        
        # 3. Range: Line length comparison
        if gen_line is not None and gt_line is not None:
            length_ratio = min(gen_line['length'], gt_line['length']) / max(gen_line['length'], gt_line['length'], 1)
            scores['range'] = length_ratio
        elif gen_line is not None:
            # Check if reasonable length
            scores['range'] = min(1.0, gen_line['length'] / 100.0)
        else:
            scores['range'] = 0.0
        
        # 4. Visual quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
        
        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
        
        scores['visual'] = red_overlap / red_union if red_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


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


class IdentifyPentagonsEvaluator(BaseEvaluator):
    """
    G-206: Identify pentagons evaluator.
    
    Rule-based evaluation:
    - Edge count identification (40%): Correct 5-sided polygon identified
    - Marking precision (35%): Red circle accurately marks pentagon
    - Marking quality (15%): Circle complete and proper
    - Scene fidelity (10%): All polygons preserved
    """
    
    TASK_WEIGHTS = {
        'edge_count': 0.40,
        'marking': 0.35,
        'quality': 0.15,
        'fidelity': 0.10
    }
    
    def _detect_polygons(self, frame: np.ndarray) -> List[Dict]:
        """Detect polygons and count their edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            polygons.append({
                'center': (cx, cy),
                'vertices': len(approx),
                'area': area,
                'is_pentagon': len(approx) == 5
            })
        
        return polygons
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate identify pentagons task."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect polygons from FIRST frame (before marking, more accurate)
        # The red marking in final frame can interfere with contour detection
        first_polygons = self._detect_polygons(first_frame) if first_frame is not None else []
        gen_polygons = self._detect_polygons(last_frame)
        gt_polygons = self._detect_polygons(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Edge count identification: Check if marking is near a pentagon
        # Use first_frame polygons for pentagon detection (more accurate)
        polygons_to_check = first_polygons if first_polygons else gen_polygons
        
        if gen_marking is not None and polygons_to_check:
            marked_pentagon = False
            for poly in polygons_to_check:
                dist = np.sqrt((gen_marking[0] - poly['center'][0])**2 + 
                              (gen_marking[1] - poly['center'][1])**2)
                if dist < 100 and poly['is_pentagon']:
                    marked_pentagon = True
                    break
            
            # If no pentagon found but marking matches GT, give credit
            if not marked_pentagon and gt_marking is not None:
                marking_dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                                      (gen_marking[1] - gt_marking[1])**2)
                if marking_dist < 30:
                    scores['edge_count'] = 0.9
                else:
                    scores['edge_count'] = 0.3
            else:
                scores['edge_count'] = 1.0 if marked_pentagon else 0.3
        else:
            scores['edge_count'] = 0.2  # Detection failed
        
        # 2. Marking precision: Compare with GT marking position
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['marking'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['marking'] = 0.2  # Detection failed
        
        # 3. Quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
        
        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
        
        scores['quality'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Scene fidelity: Compare polygon counts
        if gen_polygons and gt_polygons:
            count_ratio = min(len(gen_polygons), len(gt_polygons)) / max(len(gen_polygons), len(gt_polygons), 1)
            scores['fidelity'] = count_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


class MarkTangentPointEvaluator(BaseEvaluator):
    """
    G-222: Mark tangent point of circles evaluator.
    
    Rule-based evaluation:
    - External tangent circle pair identification (40%): Correct pair found
    - Tangent point calculation accuracy (40%): Precise point location
    - Marking position accuracy (15%): Mark centered on tangent point
    - Visual annotation quality (5%): Black circle proper
    """
    
    TASK_WEIGHTS = {
        'pair_id': 0.40,
        'calculation': 0.40,
        'position': 0.15,
        'annotation': 0.05
    }
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                                    param1=50, param2=30, minRadius=20, maxRadius=200)
        
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append({
                    'center': (i[0], i[1]),
                    'radius': i[2]
                })
        
        return detected
    
    def _find_tangent_pairs(self, circles: List[Dict]) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Find externally tangent circle pairs and their tangent points."""
        tangent_pairs = []
        
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                c1, c2 = circles[i], circles[j]
                
                # Convert to float to avoid overflow
                c1x, c1y = float(c1['center'][0]), float(c1['center'][1])
                c2x, c2y = float(c2['center'][0]), float(c2['center'][1])
                r1, r2 = float(c1['radius']), float(c2['radius'])
                
                dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                
                # Check if externally tangent (distance ≈ r1 + r2)
                expected_dist = r1 + r2
                if abs(dist - expected_dist) < 10:  # Tolerance
                    # Calculate tangent point
                    t = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                    tx = c1x + t * (c2x - c1x)
                    ty = c1y + t * (c2y - c1y)
                    
                    tangent_pairs.append((i, j, (tx, ty)))
        
        return tangent_pairs
    
    def _detect_black_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect black circle marking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Black marking
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 1000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate mark tangent point task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_black_marking(last_frame)
        gt_marking = self._detect_black_marking(gt_last)
        
        # Detect circles and find tangent pairs
        gen_circles = self._detect_circles(last_frame)
        tangent_pairs = self._find_tangent_pairs(gen_circles)
        
        # 1. Pair identification: Check if marking is near a tangent point
        if gen_marking is not None and tangent_pairs:
            near_tangent = False
            for _, _, tangent_point in tangent_pairs:
                dist = np.sqrt((gen_marking[0] - tangent_point[0])**2 + 
                              (gen_marking[1] - tangent_point[1])**2)
                if dist < 30:
                    near_tangent = True
                    break
            scores['pair_id'] = 1.0 if near_tangent else 0.3
        else:
            scores['pair_id'] = 0.2  # Detection failed
        
        # 2. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)
        else:
            scores['calculation'] = 0.2  # Detection failed
        
        # 3. Position accuracy: Same with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
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
}
