"""Evaluator for O-22_construction_stack_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-22_construction_stack_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ConstructionStackEvaluator(BaseEvaluator):
    """
    O-22: Construction stack (block stacking) evaluator.
    
    CRITICAL RULES:
    1. TARGET (right stack) should remain UNCHANGED from first to last frame
    2. RESULT (left stack in final) should match TARGET (right stack in first)
    3. SOURCE blocks (left side in first frame) should be moved to create the result
    4. Animation should show actual block movement
    
    Evaluation dimensions:
    - Target preservation (25%): Right stack unchanged
    - Final state correctness (40%): Left stack matches target pattern
    - Source changed (20%): Left side actually changed from original
    - Movement detection (15%): Visible block movement animation
    """
    
    TASK_WEIGHTS = {
        'target_preservation': 0.25,
        'final_state': 0.40,
        'source_changed': 0.20,
        'movement_detection': 0.15
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
        h, w = first_frame.shape[:2]
        
        # Detect blocks in different regions
        left_boundary = w // 3
        right_boundary = 2 * w // 3
        
        # Target region (right side)
        first_right = first_frame[:, right_boundary:]
        final_right = final_frame[:, right_boundary:]
        first_target_blocks = self._detect_all_blocks_fine_region(first_right)
        final_target_blocks = self._detect_all_blocks_fine_region(final_right)
        
        # Source region (left side) 
        first_left = first_frame[:, :left_boundary]
        final_left = final_frame[:, :left_boundary]
        first_source_blocks = self._detect_all_blocks_fine_region(first_left)
        final_result_blocks = self._detect_all_blocks_fine_region(final_left)
        
        # Store debug info
        scores['first_target_count'] = len(first_target_blocks)
        scores['final_target_count'] = len(final_target_blocks)
        scores['first_source_count'] = len(first_source_blocks)
        scores['final_result_count'] = len(final_result_blocks)
        
        # 1. Target preservation (25%): Right stack must remain unchanged
        target_preservation = self._evaluate_target_preservation_v2(
            first_target_blocks, final_target_blocks
        )
        scores['target_preservation'] = target_preservation
        
        # If target changed significantly, the task failed
        if target_preservation < 0.3:
            scores['final_state'] = 0.0
            scores['source_changed'] = 0.0
            scores['movement_detection'] = 0.0
            scores['error'] = 'target_stack_changed'
            self._last_task_details = scores
            return target_preservation * self.TASK_WEIGHTS['target_preservation']
        
        # 2. Final state (40%): Result stack should match target pattern
        final_state_score = self._evaluate_final_state_v2(
            first_target_blocks, final_result_blocks
        )
        scores['final_state'] = final_state_score
        
        # 3. Source changed (20%): Left side should have changed
        source_changed_score = self._evaluate_source_changed(
            first_source_blocks, final_result_blocks, first_target_blocks
        )
        scores['source_changed'] = source_changed_score
        
        # 4. Movement detection (15%): Visible movement in video
        movement_score = self._detect_block_movement(video_frames, left_boundary, right_boundary)
        scores['movement_detection'] = movement_score
        
        # STRICT: If no movement detected, fail
        if movement_score < 0.2:
            scores['error'] = 'no_movement_detected'
            self._last_task_details = scores
            return 0.0
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _evaluate_target_preservation_v2(self, first_target: List[Dict], 
                                         final_target: List[Dict]) -> float:
        """Check if target stack (right side) remained unchanged."""
        if len(first_target) == 0:
            return 0.0  # No target blocks = can't evaluate
        
        # Block count must be identical
        if len(first_target) != len(final_target):
            return 0.0
        
        # Sort by y position and compare colors
        first_sorted = sorted(first_target, key=lambda b: b['y'])
        final_sorted = sorted(final_target, key=lambda b: b['y'])
        
        matched = 0
        for fb, lb in zip(first_sorted, final_sorted):
            hue_diff = abs(fb['hue'] - lb['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 30:
                matched += 1
        
        return matched / len(first_target)
    
    def _evaluate_final_state_v2(self, target_blocks: List[Dict], 
                                  result_blocks: List[Dict]) -> float:
        """Check if result stack matches target pattern (colors in correct order)."""
        if len(target_blocks) == 0:
            return 0.0
        
        # STRICT: Block count must match
        if len(result_blocks) != len(target_blocks):
            count_diff = abs(len(result_blocks) - len(target_blocks))
            if count_diff == 1:
                return 0.1  # One block off
            return 0.0
        
        # Sort by y position (bottom to top)
        target_sorted = sorted(target_blocks, key=lambda b: b['y'], reverse=True)
        result_sorted = sorted(result_blocks, key=lambda b: b['y'], reverse=True)
        
        matched = 0
        for target_b, result_b in zip(target_sorted, result_sorted):
            hue_diff = abs(target_b['hue'] - result_b['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 25:
                matched += 1
        
        ratio = matched / len(target_sorted)
        
        if ratio >= 0.9:
            return 1.0
        elif ratio >= 0.7:
            return 0.6
        elif ratio >= 0.5:
            return 0.3
        else:
            return 0.0
    
    def _evaluate_source_changed(self, first_source: List[Dict], 
                                 final_result: List[Dict],
                                 target_blocks: List[Dict]) -> float:
        """Check if the source (left side) changed to create the result."""
        if len(first_source) == 0 and len(final_result) == 0:
            return 0.0  # Nothing changed
        
        # Compare colors - result should be different from original source
        # but should match target
        first_hues = sorted([b['hue'] for b in first_source])
        final_hues = sorted([b['hue'] for b in final_result])
        target_hues = sorted([b['hue'] for b in target_blocks])
        
        # Check if final matches target more than original
        target_match = self._compare_hue_lists(final_hues, target_hues)
        source_match = self._compare_hue_lists(first_hues, target_hues)
        
        if target_match > source_match:
            return 1.0  # Good: result is closer to target than source was
        elif target_match > 0.5:
            return 0.5  # Partial: result somewhat matches target
        else:
            return 0.2
    
    def _compare_hue_lists(self, hues1: List[int], hues2: List[int]) -> float:
        """Compare two sorted lists of hues."""
        if not hues1 or not hues2:
            return 0.0
        if len(hues1) != len(hues2):
            return 0.2
        
        matched = 0
        for h1, h2 in zip(hues1, hues2):
            diff = abs(h1 - h2)
            diff = min(diff, 180 - diff)
            if diff < 25:
                matched += 1
        
        return matched / len(hues1)
    
    def _detect_block_movement(self, frames: List[np.ndarray], 
                               left_boundary: int, right_boundary: int) -> float:
        """
        Detect if there's visible block movement in the video.
        
        For construction stack task, movement happens primarily in the LEFT region
        where blocks are being stacked. We check both left and middle regions.
        """
        if len(frames) < 3:
            return 0.0
        
        # Sample frames to detect movement
        sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        sample_indices = [min(i, len(frames)-1) for i in sample_indices]
        
        # Look for changes in the LEFT region (where blocks are stacked)
        left_region_changes = 0
        prev_left = None
        prev_middle = None
        
        for idx in sample_indices:
            frame = frames[idx]
            # Extract left region (where stacking happens)
            left = frame[:, :left_boundary]
            middle = frame[:, left_boundary:right_boundary]
            
            if prev_left is not None:
                left_diff = np.mean(np.abs(left.astype(float) - prev_left.astype(float)))
                middle_diff = np.mean(np.abs(middle.astype(float) - prev_middle.astype(float)))
                # Either region having significant change counts
                if left_diff > 3 or middle_diff > 3:
                    left_region_changes += 1
            
            prev_left = left.copy()
            prev_middle = middle.copy()
        
        # Score based on detected changes
        if left_region_changes >= 3:
            return 1.0
        elif left_region_changes >= 2:
            return 0.7
        elif left_region_changes >= 1:
            return 0.4
        else:
            return 0.0
    
    def _detect_all_blocks_fine_region(self, region: np.ndarray) -> List[Dict]:
        """
        Detect colored blocks in a specific region.
        
        Uses lower saturation threshold (30) to detect blocks that might be less saturated.
        Blocks in construction stack are typically small colored squares (~2000-2500 area).
        
        NOTE: When blocks are stacked and touching, they merge into a single contour.
        We estimate the number of blocks by area/expected_block_area.
        """
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions - lower threshold for better detection
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        expected_single_block_area = 2200  # Typical single block area
        min_block_area = 500  # Filter out noise smaller than this
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (blocks should be at least 500 area)
            if area < min_block_area:
                continue
            
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get average hue
            mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask_cnt)[:3]
            
            # Estimate number of blocks from area
            # A single block is ~2000-2500 area
            # If area is large, it might be multiple stacked blocks
            estimated_block_count = max(1, int(round(area / expected_single_block_area)))
            
            if estimated_block_count == 1:
                blocks.append({
                    'x': cx,
                    'y': cy,
                    'hue': int(mean_hsv[0]),
                    'area': area
                })
            else:
                # Create multiple "virtual" blocks for a stacked column
                # Distribute them vertically based on bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                block_height = h / estimated_block_count
                for i in range(estimated_block_count):
                    block_cy = int(y + (i + 0.5) * block_height)
                    blocks.append({
                        'x': cx,
                        'y': block_cy,
                        'hue': int(mean_hsv[0]),
                        'area': expected_single_block_area
                    })
        
        return blocks
    
    def _detect_all_blocks_fine(self, frame: np.ndarray) -> List[Dict]:
        """Detect all colored blocks with finer granularity (lower area threshold)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Lower threshold to catch smaller blocks, but filter out noise
            if 200 < area < 15000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color using mask
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    # Get hue
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_color = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    blocks.append({
                        'x': cx,
                        'y': cy,
                        'color': mean_color,
                        'hue': int(hsv_color[0]),
                        'area': area
                    })
        
        return blocks
    
    def _detect_blocks_with_color(self, region: np.ndarray) -> List[Dict]:
        """Detect colored blocks with center and average color."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get average color using mask
                    mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(region, mask=mask_cnt)[:3]
                    blocks.append({'center': (cx, cy), 'color': mean_color, 'area': area})
        
        return blocks
    
    def _evaluate_steps(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate if step count is reasonable."""
        # Count significant frame changes
        changes = 0
        for i in range(1, min(len(video_frames), 50)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            if diff > 5:
                changes += 1
        
        # Reasonable number of moves
        if changes < 20:
            return 1.0
        elif changes < 40:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_movement(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate movement smoothness."""
        if len(video_frames) < 3:
            return 0.5
        
        diffs = []
        for i in range(1, min(len(video_frames), 20)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            diffs.append(diff)
        
        if len(diffs) < 2:
            return 0.5
        
        variance = np.var(diffs)
        return 1.0 / (1.0 + variance / 100)
    
    def _detect_blocks(self, region: np.ndarray) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """Detect colored blocks with (x, y, color)."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    color = tuple(int(c) for c in region[cy, cx])
                    blocks.append((cx, cy, color))
        
        return blocks
