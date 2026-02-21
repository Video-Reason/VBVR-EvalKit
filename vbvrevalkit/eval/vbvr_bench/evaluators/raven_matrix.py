"""Evaluator for O-56_raven_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-56_raven_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class RavenMatrixEvaluator(BaseEvaluator):
    """
    O-56: Raven's Progressive Matrices evaluator.
    
    CRITICAL RULES:
    1. The video shows a 3x3 grid (9 cells)
    2. ONLY the bottom-right cell (position 2,2) should change
    3. The other 8 cells MUST remain UNCHANGED
    4. The answer cell should contain the correct pattern based on the rules
    
    Evaluation dimensions:
    - Other cells preserved (40%): CRITICAL - first 8 cells must not change
    - Answer cell correct (40%): Bottom-right has correct shape/pattern
    - Answer cell has content (15%): Something was drawn in answer cell
    - Grid structure preserved (5%): 3x3 grid structure maintained
    """
    
    TASK_WEIGHTS = {
        'preservation': 0.40,       # CRITICAL: other 8 cells unchanged
        'answer_correct': 0.40,     # Answer cell matches GT
        'answer_has_content': 0.15, # Answer cell is not empty
        'grid_structure': 0.05
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
        
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        scores = {}
        
        # Resize frames to match if needed
        if gt_final_frame is not None and final_frame.shape != gt_final_frame.shape:
            gt_final_frame = cv2.resize(gt_final_frame, (final_frame.shape[1], final_frame.shape[0]))
        if gt_first_frame is not None and first_frame.shape != gt_first_frame.shape:
            gt_first_frame = cv2.resize(gt_first_frame, (first_frame.shape[1], first_frame.shape[0]))
        
        # Detect what's in each cell
        first_cell_info = self._analyze_all_cells(first_frame)
        final_cell_info = self._analyze_all_cells(final_frame)
        
        # Store debug info
        scores['first_cell_counts'] = [c['count'] for c in first_cell_info]
        scores['final_cell_counts'] = [c['count'] for c in final_cell_info]
        
        # 1. CRITICAL: Check if other 8 cells are preserved (40%)
        preservation_score = self._evaluate_other_cells_preserved(
            first_frame, final_frame, first_cell_info, final_cell_info
        )
        scores['preservation'] = preservation_score
        
        # If preservation is too low, other scores are less meaningful
        if preservation_score < 0.5:
            scores['error'] = 'other_cells_changed'
        
        # 2. Check if answer cell (2,2) is correct (40%)
        if gt_final_frame is not None:
            answer_score = self._evaluate_answer_cell(final_frame, gt_final_frame)
        else:
            answer_score = 0.5  # Can't evaluate without GT
        scores['answer_correct'] = answer_score
        
        # 3. Check if answer cell has content (15%)
        answer_cell = self._extract_cell(final_frame, 2, 2)
        answer_props = self._detect_shapes_in_cell(answer_cell)
        scores['answer_has_content'] = 1.0 if answer_props['count'] > 0 else 0.0
        scores['answer_shape_count'] = answer_props['count']
        scores['answer_shape_types'] = answer_props['types']
        
        # 4. Grid structure (5%)
        scores['grid_structure'] = self._evaluate_grid_structure(final_frame)
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _analyze_all_cells(self, frame: np.ndarray) -> List[Dict]:
        """Analyze shapes in all 9 cells."""
        cells = []
        for row in range(3):
            for col in range(3):
                cell = self._extract_cell(frame, row, col)
                props = self._detect_shapes_in_cell(cell)
                props['row'] = row
                props['col'] = col
                cells.append(props)
        return cells
    
    def _evaluate_other_cells_preserved(self, first_frame: np.ndarray, final_frame: np.ndarray,
                                        first_info: List[Dict], final_info: List[Dict]) -> float:
        """CRITICAL: Check that all 8 cells except bottom-right are unchanged."""
        unchanged_count = 0
        total_checked = 0
        
        for row in range(3):
            for col in range(3):
                if row == 2 and col == 2:
                    continue  # Skip answer cell
                
                idx = row * 3 + col
                first_props = first_info[idx]
                final_props = final_info[idx]
                
                total_checked += 1
                
                # Check if cell is unchanged
                # 1. Same shape count
                if first_props['count'] != final_props['count']:
                    continue
                
                # 2. Same shape types
                if sorted(first_props['types']) != sorted(final_props['types']):
                    continue
                
                # 3. Similar pixel content (using cell comparison)
                first_cell = self._extract_cell(first_frame, row, col)
                final_cell = self._extract_cell(final_frame, row, col)
                
                # Compare cells
                diff = np.mean(np.abs(first_cell.astype(float) - final_cell.astype(float)))
                if diff < 15:  # Very similar
                    unchanged_count += 1
                elif diff < 30:  # Somewhat similar (partial credit)
                    unchanged_count += 0.5
        
        return unchanged_count / total_checked if total_checked > 0 else 0.0
    
    def _evaluate_answer_cell(self, final: np.ndarray, gt_final: np.ndarray) -> float:
        """Check if answer cell (bottom-right) matches GT."""
        final_cell = self._extract_cell(final, 2, 2)
        gt_cell = self._extract_cell(gt_final, 2, 2)
        
        final_props = self._detect_shapes_in_cell(final_cell)
        gt_props = self._detect_shapes_in_cell(gt_cell)
        
        score = 0.0
        
        # Check shape count (most important)
        if final_props['count'] == gt_props['count']:
            score += 0.5
        elif abs(final_props['count'] - gt_props['count']) == 1:
            score += 0.2
        
        # Check shape types
        if sorted(final_props['types']) == sorted(gt_props['types']):
            score += 0.3
        elif set(final_props['types']) & set(gt_props['types']):
            score += 0.1
        
        # Check fill patterns
        if final_props['filled'] == gt_props['filled']:
            score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_grid_structure(self, frame: np.ndarray) -> float:
        """Check if 3x3 grid structure is preserved."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = gray.shape
        
        # Check for horizontal and vertical lines at grid boundaries
        grid_score = 0.0
        
        # Check horizontal lines at 1/3 and 2/3 height
        for y_frac in [1/3, 2/3]:
            y = int(h * y_frac)
            line_region = edges[max(0, y-5):min(h, y+5), :]
            if np.sum(line_region > 0) > w * 0.3:  # At least 30% of width has edges
                grid_score += 0.25
        
        # Check vertical lines at 1/3 and 2/3 width
        for x_frac in [1/3, 2/3]:
            x = int(w * x_frac)
            line_region = edges[:, max(0, x-5):min(w, x+5)]
            if np.sum(line_region > 0) > h * 0.3:  # At least 30% of height has edges
                grid_score += 0.25
        
        return grid_score
    
    def _extract_cell(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a single cell from the 3x3 matrix."""
        h, w = frame.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        return frame[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
    
    def _detect_shapes_in_cell(self, cell: np.ndarray) -> Dict:
        """Detect shapes in a cell and return properties."""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            # Approximate contour to get shape type
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)
            
            # Determine shape type
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'square'
            elif vertices >= 6:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if circularity > 0.7:
                    shape_type = 'circle'
                else:
                    shape_type = 'polygon'
            else:
                shape_type = 'other'
            
            # Check if filled
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Sample inside
                if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                    is_filled = gray[cy, cx] < 128
                else:
                    is_filled = False
            else:
                is_filled = False
            
            shapes.append({
                'type': shape_type,
                'area': area,
                'vertices': vertices,
                'filled': is_filled
            })
        
        return {
            'count': len(shapes),
            'shapes': shapes,
            'types': [s['type'] for s in shapes],
            'filled': [s['filled'] for s in shapes]
        }
