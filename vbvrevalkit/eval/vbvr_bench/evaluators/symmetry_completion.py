"""Evaluator for O-49_symmetry_completion_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-49_symmetry_completion_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class SymmetryCompletionEvaluator(BaseEvaluator):
    """
    O-49: Symmetry completion evaluator.
    
    Rule-based evaluation:
    - Block preservation (40%): Original blocks unchanged, total count correct
    - Symmetry accuracy (35%): Filled blocks create left-right symmetry
    - Fill correctness (20%): Blocks filled at correct symmetric positions
    - Color consistency (5%): New blocks match original block colors
    """
    
    TASK_WEIGHTS = {
        'block_preservation': 0.40,
        'symmetry_accuracy': 0.35,
        'fill_correctness': 0.20,
        'color_consistency': 0.05
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
        
        # Detect filled blocks
        first_blocks = self._detect_filled_blocks(first_frame)
        final_blocks = self._detect_filled_blocks(final_frame)
        
        # 1. Block preservation - original blocks should remain, correct total count
        scores['block_preservation'] = self._evaluate_block_preservation(
            first_blocks, final_blocks, gt_first_frame, gt_final_frame
        )
        
        # If blocks are completely changed, penalize heavily
        if scores['block_preservation'] < 0.3:
            self._last_task_details = {
                'block_preservation': scores['block_preservation'],
                'symmetry_accuracy': 0.0,
                'fill_correctness': 0.0,
                'color_consistency': 0.0,
                'blocks_changed': True
            }
            return scores['block_preservation'] * self.TASK_WEIGHTS['block_preservation']
        
        # 2. Symmetry accuracy
        scores['symmetry_accuracy'] = self._evaluate_symmetry(final_blocks, final_frame)
        
        # 3. Fill correctness
        scores['fill_correctness'] = self._evaluate_fill_correctness(
            first_blocks, final_blocks, final_frame
        )
        
        # 4. Color consistency
        scores['color_consistency'] = self._evaluate_color_consistency(
            first_blocks, final_blocks
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_filled_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect filled (colored) blocks."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 50000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # Get color
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    blocks.append({
                        'center': (cx, cy),
                        'color': mean_color,
                        'hue': int(hsv_c[0]),
                        'area': area
                    })
        return blocks
    
    def _evaluate_block_preservation(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray]
    ) -> float:
        """Check if original blocks are preserved and total count is correct."""
        first_count = len(first_blocks)
        final_count = len(final_blocks)
        
        # Get expected final count from GT if available
        if gt_final_frame is not None:
            gt_final_blocks = self._detect_filled_blocks(gt_final_frame)
            expected_count = len(gt_final_blocks)
        else:
            # Estimate: should add blocks to make symmetric
            expected_count = first_count + (first_count // 2)  # Rough estimate
        
        # Check count
        if final_count < first_count:
            return 0.0  # Blocks were removed - bad
        
        count_diff = abs(final_count - expected_count)
        if count_diff == 0:
            count_score = 1.0
        elif count_diff <= 2:
            count_score = 0.7
        else:
            count_score = 0.3
        
        # Check if original blocks' colors are preserved
        first_hues = sorted([b['hue'] for b in first_blocks])
        
        # Find matching hues in final
        matched = 0
        final_hues = [b['hue'] for b in final_blocks]
        used = set()
        for fh in first_hues:
            for i, fnlh in enumerate(final_hues):
                if i in used:
                    continue
                hue_diff = abs(fh - fnlh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    matched += 1
                    used.add(i)
                    break
        
        color_preservation = matched / len(first_hues) if first_hues else 0.0
        
        return (count_score + color_preservation) / 2
    
    def _evaluate_symmetry(self, final_blocks: List[Dict], final_frame: np.ndarray) -> float:
        """Check if blocks form left-right symmetry."""
        if len(final_blocks) == 0:
            return 0.0
        
        h, w = final_frame.shape[:2]
        center_x = w // 2
        
        # Group blocks by their y-coordinate (rows)
        rows = {}
        for block in final_blocks:
            y = block['center'][1]
            # Quantize y to group nearby blocks
            row_key = y // 40
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(block)
        
        # For each row, check if blocks are symmetric around center
        symmetric_rows = 0
        total_rows = len(rows)
        
        for row_key, row_blocks in rows.items():
            # Get x positions relative to center
            x_positions = [b['center'][0] - center_x for b in row_blocks]
            
            # Check if for each x, there's a -x
            is_symmetric = True
            for x in x_positions:
                if x == 0:
                    continue  # Center block
                has_mirror = any(abs(x + other_x) < 30 for other_x in x_positions)
                if not has_mirror:
                    is_symmetric = False
                    break
            
            if is_symmetric:
                symmetric_rows += 1
        
        return symmetric_rows / total_rows if total_rows > 0 else 0.0
    
    def _evaluate_fill_correctness(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict],
        final_frame: np.ndarray
    ) -> float:
        """Check if new blocks are filled at correct symmetric positions."""
        h, w = final_frame.shape[:2]
        center_x = w // 2
        
        # Find new blocks (in final but not in first)
        first_positions = set()
        for b in first_blocks:
            # Quantize position
            pos_key = (b['center'][0] // 30, b['center'][1] // 30)
            first_positions.add(pos_key)
        
        new_blocks = []
        for b in final_blocks:
            pos_key = (b['center'][0] // 30, b['center'][1] // 30)
            if pos_key not in first_positions:
                new_blocks.append(b)
        
        if len(new_blocks) == 0:
            return 0.5  # No new blocks - might be okay if already symmetric
        
        # Check if each new block has a corresponding original block on the other side
        correct_fills = 0
        for new_block in new_blocks:
            new_x = new_block['center'][0]
            new_y = new_block['center'][1]
            
            # Calculate mirror position
            mirror_x = 2 * center_x - new_x
            
            # Check if there's an original block at the mirror position
            has_mirror = False
            for orig_block in first_blocks:
                orig_x = orig_block['center'][0]
                orig_y = orig_block['center'][1]
                if abs(orig_x - mirror_x) < 30 and abs(orig_y - new_y) < 30:
                    has_mirror = True
                    break
            
            if has_mirror:
                correct_fills += 1
        
        return correct_fills / len(new_blocks) if new_blocks else 0.5
    
    def _evaluate_color_consistency(
        self, 
        first_blocks: List[Dict], 
        final_blocks: List[Dict]
    ) -> float:
        """Check if new blocks use consistent colors."""
        first_hues = set(b['hue'] for b in first_blocks)
        final_hues = [b['hue'] for b in final_blocks]
        
        # Check how many final block hues are similar to first block hues
        consistent = 0
        for fh in final_hues:
            for oh in first_hues:
                hue_diff = abs(fh - oh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    consistent += 1
                    break
        
        return consistent / len(final_hues) if final_hues else 0.5
        
        return max(0.0, correlation)
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if left side is preserved."""
        h, w = first_frame.shape[:2]
        
        first_left = first_frame[:, :w//2]
        final_left = final_frame[:, :w//2]
        
        # Compare
        diff = np.mean(np.abs(first_left.astype(float) - final_left.astype(float)))
        
        if diff < 10:
            return 1.0
        elif diff < 30:
            return 0.7
        else:
            return 0.4
    
    def _count_filled_cells(self, region: np.ndarray) -> int:
        """Count filled (dark) cells."""
        if region.size == 0:
            return 0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return np.sum(gray < 100)


# Export all Part 3 evaluators
HIDDEN40_EVALUATORS_PART3 = {
    'O-2_pigment_color_mixing_subtractive_data-generator': PigmentColorMixingEvaluator,
    'O-5_symbol_deletion_data-generator': SymbolDeletionEvaluator,
    'O-6_2d_geometric_transformation_data-generator': GeometricTransformationEvaluator,
    'O-9_shape_scaling_data-generator': ShapeScalingAnalogyEvaluator,
    'O-11_shape_color_then_move_data-generator': ShapeColorThenMoveEvaluator,
    'O-22_construction_stack_data-generator': ConstructionStackEvaluator,
    'O-39_maze_data-generator': MazePathfindingEvaluator,
    'O-43_object_subtraction_data-generator': ObjectSubtractionEvaluator,
    'O-46_shape_sorter_data-generator': ShapeSorterEvaluator,
    'O-49_symmetry_completion_data-generator': SymmetryCompletionEvaluator,
