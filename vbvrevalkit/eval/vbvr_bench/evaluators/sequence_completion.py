"""Evaluator for O-45_sequence_completion_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-45_sequence_completion_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class SequenceCompletionEvaluator(BaseEvaluator):
    """
    O-45: Sequence Completion
    
    Task: Observe pattern in sequence (numbers/shapes/colors/directions) 
    and replace ? with correct next element.
    
    Key evaluation criteria:
    1. Sequence type identification (35%) - Correct pattern type
    2. Element calculation (35%) - Correct value computed
    3. Element rendering (20%) - Visual consistency
    4. Sequence integrity (10%) - Complete sequence valid
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'sequence_type_identification': 0.35,
            'element_calculation': 0.35,
            'element_rendering': 0.20,
            'sequence_integrity': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate sequence completion accuracy - RULE-BASED."""
        
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        gt_first = gt_first_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gen_final = cv2.resize(gen_final, (gt_final.shape[1], gt_final.shape[0]))
        
        scores = {}
        
        # RULE-BASED: Detect sequence elements and check if answer is correct
        # Step 1: Detect elements in GT first (the pattern)
        gt_first_elements = self._detect_sequence_elements(gt_first)
        
        # Step 2: Detect elements in GT final (includes correct answer)
        gt_final_elements = self._detect_sequence_elements(gt_final)
        
        # Step 3: Detect elements in generated final
        gen_final_elements = self._detect_sequence_elements(gen_final)
        
        # The answer should be the rightmost element in GT final
        # that wasn't in GT first (or the ? position)
        gt_answer_color = None
        if len(gt_final_elements) > len(gt_first_elements):
            gt_answer_color = gt_final_elements[-1][2]  # Color of last element
        elif len(gt_final_elements) > 0:
            gt_answer_color = gt_final_elements[-1][2]
        
        # Check if generated has the correct answer
        gen_answer_color = None
        answer_added = len(gen_final_elements) >= len(gt_final_elements)
        if len(gen_final_elements) > 0:
            gen_answer_color = gen_final_elements[-1][2]
        
        # 1. Sequence type identification (35%): Did they add an element?
        if not answer_added:
            scores['sequence_type_identification'] = 0.0  # No answer added
        elif gt_answer_color is not None and gen_answer_color is not None:
            # Check if answer color matches
            color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(gt_answer_color, gen_answer_color)))
            if color_diff < 50:
                scores['sequence_type_identification'] = 1.0
            elif color_diff < 100:
                scores['sequence_type_identification'] = 0.3
            else:
                scores['sequence_type_identification'] = 0.0
        else:
            scores['sequence_type_identification'] = 0.0
        
        # 2. Element calculation (35%): Is the answer color correct?
        if not answer_added:
            scores['element_calculation'] = 0.0
        elif gt_answer_color is not None and gen_answer_color is not None:
            color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(gt_answer_color, gen_answer_color)))
            if color_diff < 50:
                scores['element_calculation'] = 1.0
            elif color_diff < 100:
                scores['element_calculation'] = 0.3
            else:
                scores['element_calculation'] = 0.0
        else:
            scores['element_calculation'] = 0.0
        
        # 3. Element rendering (20%): Is the element count correct?
        if len(gen_final_elements) == len(gt_final_elements):
            scores['element_rendering'] = 1.0
        elif abs(len(gen_final_elements) - len(gt_final_elements)) == 1:
            scores['element_rendering'] = 0.5
        else:
            scores['element_rendering'] = 0.0
        
        # 4. Sequence integrity (10%): Are original elements preserved?
        if len(gt_first_elements) > 0:
            # Check if first N-1 elements match
            preserved = 0
            for i, (_, _, gt_color) in enumerate(gt_first_elements[:-1]):  # Exclude ? position
                if i < len(gen_final_elements):
                    gen_color = gen_final_elements[i][2]
                    color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(gt_color, gen_color)))
                    if color_diff < 80:
                        preserved += 1
            
            if len(gt_first_elements) > 1:
                scores['sequence_integrity'] = preserved / (len(gt_first_elements) - 1)
            else:
                scores['sequence_integrity'] = 1.0
        else:
            scores['sequence_integrity'] = 0.0  # STRICT: No GT elements
        
        self._last_task_details = scores
        self._last_task_details['gt_first_count'] = len(gt_first_elements)
        self._last_task_details['gt_final_count'] = len(gt_final_elements)
        self._last_task_details['gen_final_count'] = len(gen_final_elements)
        
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_sequence_elements(self, frame: np.ndarray) -> List[Tuple[int, int, Tuple]]:
        """Detect sequence elements (colored shapes OR numbers) with their colors."""
        elements = []
        
        # Method 1: Try detecting saturated (colored) regions first
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter noise
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Get dominant color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    
                    elements.append((cx, cy, mean_color))
        
        # Method 2: If no colored elements found, try detecting dark elements (numbers/text)
        if len(elements) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Group nearby contours (digits of same number may be separate)
            # For simplicity, just count distinct x-regions
            x_positions = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Filter noise
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        x_positions.append(cx)
            
            if x_positions:
                # Group x positions that are close together (same element)
                x_positions.sort()
                groups = []
                current_group = [x_positions[0]]
                
                for x in x_positions[1:]:
                    if x - current_group[-1] < 50:  # Same element
                        current_group.append(x)
                    else:  # New element
                        groups.append(current_group)
                        current_group = [x]
                groups.append(current_group)
                
                # Create elements from groups
                for group in groups:
                    cx = int(np.mean(group))
                    cy = frame.shape[0] // 2  # Assume middle y
                    # Use black color for text/numbers
                    elements.append((cx, cy, (0, 0, 0)))
        
        # Sort by x position (left to right)
        elements.sort(key=lambda e: e[0])
        return elements
    
    def _evaluate_answer_match(self, gen_answer: np.ndarray, gt_answer: np.ndarray) -> float:
        """Evaluate if answer region matches GT."""
        # Compare color histograms
        gen_hist = cv2.calcHist([gen_answer], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        gt_hist = cv2.calcHist([gt_answer], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(gen_hist, gen_hist)
        cv2.normalize(gt_hist, gt_hist)
        
        similarity = cv2.compareHist(gen_hist, gt_hist, cv2.HISTCMP_CORREL)
        
        return max(0, similarity)
    
    def _evaluate_element_calculation(self, gen_answer: np.ndarray, gt_answer: np.ndarray) -> float:
        """Evaluate if the computed element is correct."""
        # Detect dominant colors in answer regions
        gen_colors = self._get_dominant_color(gen_answer)
        gt_colors = self._get_dominant_color(gt_answer)
        
        if gen_colors is None or gt_colors is None:
            return 0.5
        
        # Compare colors
        color_diff = np.sqrt(np.sum((np.array(gen_colors) - np.array(gt_colors))**2))
        
        if color_diff < 30:
            return 1.0
        elif color_diff < 60:
            return 0.7
        elif color_diff < 100:
            return 0.4
        else:
            return 0.2
    
    def _get_dominant_color(self, region: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Get dominant non-white color in region."""
        # Exclude white/near-white pixels
        mask = np.all(region < 240, axis=2)
        
        if np.sum(mask) < 100:
            return None
        
        colored_pixels = region[mask]
        if len(colored_pixels) == 0:
            return None
        
        # Get average color
        avg_color = np.mean(colored_pixels, axis=0)
        return tuple(avg_color.astype(int))
    
    def _evaluate_element_rendering(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate visual consistency of the answer element."""
        h, w = gen_frame.shape[:2]
        
        # Compare answer region sizes
        gen_answer = gen_frame[:, w*3//4:]
        gt_answer = gt_frame[:, w*3//4:]
        
        # Detect shapes in answer regions
        gen_shapes = self._detect_shapes(gen_answer)
        gt_shapes = self._detect_shapes(gt_answer)
        
        if not gt_shapes:
            return 0.5
        
        if not gen_shapes:
            return 0.0
        
        # Compare shape sizes
        gen_areas = [s['area'] for s in gen_shapes]
        gt_areas = [s['area'] for s in gt_shapes]
        
        if gen_areas and gt_areas:
            gen_avg = np.mean(gen_areas)
            gt_avg = np.mean(gt_areas)
            
            if gt_avg > 0:
                ratio = gen_avg / gt_avg
                if 0.7 <= ratio <= 1.3:
                    return 1.0
                elif 0.5 <= ratio <= 1.5:
                    return 0.7
                else:
                    return 0.3
        
        return 0.5
    
    def _detect_shapes(self, region: np.ndarray) -> List[Dict]:
        """Detect shapes in a region."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                shapes.append({'contour': contour, 'area': area})
        
        return shapes
    
    def _evaluate_sequence_integrity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if the complete sequence is valid."""
        # Count elements in both frames
        gen_shapes = self._detect_shapes(gen_frame)
        gt_shapes = self._detect_shapes(gt_frame)
        
        if not gt_shapes:
            return 0.5
        
        # Compare counts
        ratio = len(gen_shapes) / len(gt_shapes) if gt_shapes else 0
        
        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.7 <= ratio <= 1.3:
            return 0.7
        else:
            return max(0.2, ratio if ratio < 1 else 2 - ratio)
