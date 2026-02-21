"""Evaluator for O-36_grid_shift_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-36_grid_shift_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class GridShiftEvaluator(BaseEvaluator):
    """
    O-36: Grid Shift
    
    Task: Move all colored blocks in NxN grid simultaneously in specified 
    direction (up/down/left/right) by specified steps.
    
    Key evaluation criteria:
    1. Direction correctness (30%) - All blocks move correct direction
    2. Step accuracy (30%) - Exact number of steps moved
    3. Synchronization (20%) - All blocks move together
    4. Position precision (15%) - Final positions correct
    5. Completeness (5%) - All blocks moved, properties preserved
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'direction_correctness': 0.30,
            'step_accuracy': 0.30,
            'synchronization': 0.20,
            'position_precision': 0.15,
            'completeness': 0.05
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid shift movement."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        # Detect colored blocks in first and final frames
        first_blocks = self._detect_colored_blocks(first_frame)
        gen_final_blocks = self._detect_colored_blocks(gen_final)
        gt_final_blocks = self._detect_colored_blocks(gt_final)
        
        scores = {}
        
        # CRITICAL: First check if blocks are preserved (completeness)
        # If blocks change, the whole task fails
        completeness_score = self._evaluate_completeness(first_blocks, gen_final_blocks)
        
        # Also check pattern preservation
        pattern_score = self._evaluate_block_pattern_preservation(
            first_frame, gen_final, first_blocks, gen_final_blocks
        )
        
        # Combine: blocks must be preserved AND patterns must be unchanged
        block_preserved = min(completeness_score, pattern_score) > 0.5
        scores['completeness'] = min(completeness_score, pattern_score)
        
        # If blocks are NOT preserved, all other scores should be 0
        if not block_preserved:
            scores['direction_correctness'] = 0.0
            scores['step_accuracy'] = 0.0
            scores['synchronization'] = 0.0
            scores['position_precision'] = 0.0
        else:
            # 1. Direction correctness (30%): Check if blocks moved in correct direction
            direction_score = self._evaluate_direction(first_blocks, gen_final_blocks, gt_final_blocks)
            scores['direction_correctness'] = direction_score
            
            # 2. Step accuracy (30%): Check if blocks moved correct number of steps
            step_score = self._evaluate_step_accuracy(first_blocks, gen_final_blocks, gt_final_blocks, gen_final)
            scores['step_accuracy'] = step_score
            
            # 3. Synchronization (20%): Check if all blocks moved together
            sync_score = self._evaluate_synchronization(video_frames)
            scores['synchronization'] = sync_score
            
            # 4. Position precision (15%): Check final block positions
            position_score = self._evaluate_position_precision(gen_final_blocks, gt_final_blocks)
            scores['position_precision'] = position_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_colored_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored blocks in the frame."""
        blocks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define color ranges for common block colors (lower saturation threshold)
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255], [160, 50, 50], [180, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255], None, None),
            'blue': ([100, 50, 50], [130, 255, 255], None, None),
            'yellow': ([20, 50, 50], [35, 255, 255], None, None),
            'orange': ([10, 50, 50], [20, 255, 255], None, None),
            'purple': ([130, 50, 50], [160, 255, 255], None, None),
            'cyan': ([85, 50, 50], [100, 255, 255], None, None),
        }
        
        detected_centers = set()  # Avoid duplicates
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200:  # Filter noise
                    continue
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Avoid duplicates
                    center_key = (cx // 20, cy // 20)
                    if center_key in detected_centers:
                        continue
                    detected_centers.add(center_key)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    blocks.append({
                        'color': color_name,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        # Also detect gray/neutral blocks (low saturation, medium value)
        if not blocks:
            # Look for non-white, non-black regions
            non_white = ((gray > 50) & (gray < 220)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter noise and background
                    continue
                
                # Check if roughly square (block-like)
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 10
                if aspect > 2:  # Not square enough
                    continue
                
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    blocks.append({
                        'color': 'gray',
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return blocks
    
    def _evaluate_direction(self, first_blocks: List[Dict], gen_blocks: List[Dict],
                            gt_blocks: List[Dict]) -> float:
        """Evaluate if blocks moved in correct direction."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0
        
        # Calculate expected movement direction from GT
        gt_movements = []
        for fb in first_blocks:
            # Find matching GT block by color
            for gtb in gt_blocks:
                if fb['color'] == gtb['color']:
                    dx = float(gtb['center'][0]) - float(fb['center'][0])
                    dy = float(gtb['center'][1]) - float(fb['center'][1])
                    gt_movements.append((dx, dy))
                    break
        
        if not gt_movements:
            return 0.5
        
        # Determine expected direction
        avg_dx = np.mean([m[0] for m in gt_movements])
        avg_dy = np.mean([m[1] for m in gt_movements])
        
        # Calculate actual movement
        gen_movements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb['color'] == gb['color']:
                    dx = float(gb['center'][0]) - float(fb['center'][0])
                    dy = float(gb['center'][1]) - float(fb['center'][1])
                    gen_movements.append((dx, dy))
                    break
        
        if not gen_movements:
            return 0.0
        
        actual_dx = np.mean([m[0] for m in gen_movements])
        actual_dy = np.mean([m[1] for m in gen_movements])
        
        # Check direction match
        direction_match = 0.0
        
        # Check horizontal direction
        if avg_dx != 0:
            if np.sign(actual_dx) == np.sign(avg_dx):
                direction_match += 0.5
        else:
            if abs(actual_dx) < 10:  # No horizontal movement expected
                direction_match += 0.5
        
        # Check vertical direction
        if avg_dy != 0:
            if np.sign(actual_dy) == np.sign(avg_dy):
                direction_match += 0.5
        else:
            if abs(actual_dy) < 10:  # No vertical movement expected
                direction_match += 0.5
        
        return direction_match
    
    def _evaluate_step_accuracy(self, first_blocks: List[Dict], gen_blocks: List[Dict],
                                gt_blocks: List[Dict], frame: np.ndarray) -> float:
        """Evaluate if blocks moved correct number of steps."""
        if not first_blocks or not gen_blocks or not gt_blocks:
            return 0.0
        
        # Estimate grid cell size
        h, w = frame.shape[:2]
        # Assume 4-12 grid, estimate cell size
        estimated_cell_size = w / 8  # Average estimate
        
        # Calculate expected displacement from GT
        gt_displacements = []
        for fb in first_blocks:
            for gtb in gt_blocks:
                if fb['color'] == gtb['color']:
                    dx = abs(float(gtb['center'][0]) - float(fb['center'][0]))
                    dy = abs(float(gtb['center'][1]) - float(fb['center'][1]))
                    gt_displacements.append(max(dx, dy))
                    break
        
        # Calculate actual displacement
        gen_displacements = []
        for fb in first_blocks:
            for gb in gen_blocks:
                if fb['color'] == gb['color']:
                    dx = abs(float(gb['center'][0]) - float(fb['center'][0]))
                    dy = abs(float(gb['center'][1]) - float(fb['center'][1]))
                    gen_displacements.append(max(dx, dy))
                    break
        
        if not gt_displacements or not gen_displacements:
            return 0.0
        
        avg_gt_disp = np.mean(gt_displacements)
        avg_gen_disp = np.mean(gen_displacements)
        
        if avg_gt_disp < 1:
            return 1.0 if avg_gen_disp < estimated_cell_size * 0.5 else 0.5
        
        # Calculate step difference
        ratio = avg_gen_disp / avg_gt_disp
        
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif 0.5 <= ratio <= 1.5:
            return 0.7
        elif 0.3 <= ratio <= 2.0:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_synchronization(self, frames: List[np.ndarray]) -> float:
        """Check if all blocks move synchronously."""
        if len(frames) < 3:
            return 0.5
        
        # Track block positions through video
        n_samples = min(10, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
        
        all_positions = []
        for idx in sample_indices:
            blocks = self._detect_colored_blocks(frames[idx])
            if blocks:
                positions = [b['center'] for b in blocks]
                all_positions.append(positions)
        
        if len(all_positions) < 3:
            return 0.5
        
        # Check if all blocks move together (similar displacement at each frame)
        sync_scores = []
        for i in range(1, len(all_positions)):
            if len(all_positions[i]) != len(all_positions[i-1]):
                continue
            
            displacements = []
            for j in range(len(all_positions[i])):
                dx = all_positions[i][j][0] - all_positions[i-1][j][0]
                dy = all_positions[i][j][1] - all_positions[i-1][j][1]
                displacements.append((dx, dy))
            
            if len(displacements) > 1:
                # Check variance in displacements
                dx_var = np.var([d[0] for d in displacements])
                dy_var = np.var([d[1] for d in displacements])
                
                # Low variance means synchronized movement
                max_var = max(dx_var, dy_var)
                if max_var < 100:
                    sync_scores.append(1.0)
                elif max_var < 500:
                    sync_scores.append(0.7)
                else:
                    sync_scores.append(0.3)
        
        return np.mean(sync_scores) if sync_scores else 0.5
    
    def _evaluate_position_precision(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate final position accuracy."""
        if not gen_blocks or not gt_blocks:
            return 0.0
        
        matched_scores = []
        
        for gtb in gt_blocks:
            best_dist = float('inf')
            for gb in gen_blocks:
                if gb['color'] == gtb['color']:
                    dist = safe_distance(gb['center'], gtb['center'])
                    best_dist = min(best_dist, dist)
            
            if best_dist < float('inf'):
                # Score based on distance
                if best_dist < 10:
                    matched_scores.append(1.0)
                elif best_dist < 30:
                    matched_scores.append(0.8)
                elif best_dist < 50:
                    matched_scores.append(0.5)
                else:
                    matched_scores.append(max(0.1, 1.0 - best_dist / 100))
        
        return np.mean(matched_scores) if matched_scores else 0.0
    
    def _evaluate_completeness(self, first_blocks: List[Dict], gen_blocks: List[Dict]) -> float:
        """Evaluate if all blocks are preserved with same colors."""
        if not first_blocks:
            return 0.0
        
        if not gen_blocks:
            return 0.0
        
        # Check if same number of blocks
        if len(gen_blocks) != len(first_blocks):
            return 0.0  # Block count changed - STRICT failure
        
        # Check if all block colors are preserved
        first_colors = sorted([b['color'] for b in first_blocks])
        gen_colors = sorted([b['color'] for b in gen_blocks])
        
        if first_colors != gen_colors:
            return 0.0  # Block colors changed - STRICT failure
        
        return 1.0  # All blocks preserved with same colors
    
    def _evaluate_block_pattern_preservation(
        self, 
        first_frame: np.ndarray, 
        gen_final: np.ndarray,
        first_blocks: List[Dict],
        gen_blocks: List[Dict]
    ) -> float:
        """Check if block patterns/content remain unchanged during shift."""
        if not first_blocks or not gen_blocks:
            return 0.0
        
        # For each block in first frame, extract its appearance
        # and compare with corresponding block in final frame
        preservation_scores = []
        
        for fb in first_blocks:
            # Find matching block by color in gen_blocks
            matching_gb = None
            for gb in gen_blocks:
                if gb['color'] == fb['color']:
                    matching_gb = gb
                    break
            
            if matching_gb is None:
                preservation_scores.append(0.0)
                continue
            
            # Extract block regions
            fx, fy, fw, fh = fb['bbox']
            gx, gy, gw, gh = matching_gb['bbox']
            
            # Get block regions
            first_region = first_frame[fy:fy+fh, fx:fx+fw]
            gen_region = gen_final[gy:gy+gh, gx:gx+gw]
            
            # Resize to same size for comparison
            if first_region.size > 0 and gen_region.size > 0:
                target_size = (max(fw, gw), max(fh, gh))
                first_resized = cv2.resize(first_region, target_size)
                gen_resized = cv2.resize(gen_region, target_size)
                
                # Compare patterns
                diff = np.abs(first_resized.astype(float) - gen_resized.astype(float)).mean()
                
                if diff < 30:  # Very similar
                    preservation_scores.append(1.0)
                elif diff < 60:
                    preservation_scores.append(0.5)
                else:
                    preservation_scores.append(0.0)  # Pattern changed
            else:
                preservation_scores.append(0.0)
        
        return np.mean(preservation_scores) if preservation_scores else 0.0
