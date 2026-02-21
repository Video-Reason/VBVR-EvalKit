"""Evaluator for O-75_communicating_vessels_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-75_communicating_vessels_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class CommunicatingVesselsEvaluator(BaseEvaluator):
    """
    O-75: Communicating Vessels
    
    Task: Simulate fluid flow in connected vessels until hydrostatic 
    equilibrium (all levels equal).
    
    Key evaluation criteria:
    1. Final equilibrium (40%) - All levels equal (average)
    2. Flow process (30%) - Realistic exponential decay
    3. Volume conservation (20%) - Total volume unchanged
    4. Visual fidelity (10%) - Vessels and markings preserved
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'final_equilibrium': 0.40,
            'flow_process': 0.30,
            'volume_conservation': 0.20,
            'visual_fidelity': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate communicating vessels simulation.
        
        CRITICAL RULE: The final frame MUST show equilibrium (all liquid levels equal).
        Compare generated video's final frame against GT final_frame.png (target state).
        """
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame  # This is the target equilibrium state
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # Get GT target levels (from gt_final_frame which shows equilibrium)
        gt_levels = self._detect_liquid_levels(gt_final)
        gen_levels = self._detect_liquid_levels(gen_final)
        
        # 1. Final equilibrium (40%): Check if liquid levels match GT equilibrium
        # CRITICAL: Compare against GT target, not just check if levels are equal
        equilibrium_score = self._evaluate_final_equilibrium_vs_gt(gen_levels, gt_levels)
        scores['final_equilibrium'] = equilibrium_score
        
        # CRITICAL: If equilibrium is not reached, heavily penalize other scores
        if equilibrium_score < 0.5:
            # Task fundamentally failed - liquid levels don't match GT
            scores['flow_process'] = 0.3
            scores['volume_conservation'] = 0.3
            scores['visual_fidelity'] = 0.5
            self._last_task_details = scores
            self._last_task_details['equilibrium_not_reached'] = True
            return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
        
        # 2. Flow process (30%): Check realistic flow
        flow_score = self._evaluate_flow_process(video_frames)
        frame_diff = cv2.absdiff(video_frames[0], video_frames[-1])
        mean_diff = np.mean(frame_diff)
        if mean_diff < 10:
            scores['flow_process'] = max(flow_score, 0.8)
        else:
            scores['flow_process'] = flow_score
        
        # 3. Volume conservation (20%): Check total volume
        conservation_score = self._evaluate_volume_conservation(first_frame, gen_final)
        scores['volume_conservation'] = conservation_score
        
        # 4. Visual fidelity (10%): Check vessel structure
        fidelity_score = self._evaluate_visual_fidelity(gen_final, gt_final)
        scores['visual_fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _evaluate_final_equilibrium_vs_gt(self, gen_levels: List[int], gt_levels: List[int]) -> float:
        """Compare generated levels against GT target levels.
        
        CRITICAL: GT levels are the target equilibrium state (all equal).
        Generated levels should match GT levels (within tolerance).
        
        The key criterion is that ALL liquid levels should be at the SAME y-coordinate.
        """
        if len(gen_levels) < 2 or len(gt_levels) < 2:
            return 0.2
        
        # GT levels should be approximately equal (equilibrium)
        gt_mean = np.mean(gt_levels)
        
        # Generated levels should also be approximately equal
        gen_std = np.std(gen_levels)
        gen_range = max(gen_levels) - min(gen_levels)
        gen_mean = np.mean(gen_levels)
        
        # CRITICAL: Check if generated levels are at equilibrium (all same y)
        # Strict threshold: levels must be within 15 pixels of each other
        if gen_range <= 15:
            # Good equilibrium - check if mean matches GT
            level_diff = abs(gen_mean - gt_mean)
            if level_diff < 30:
                return 1.0
            elif level_diff < 60:
                return 0.9
            else:
                return 0.7  # Equilibrium reached but at different level
        elif gen_range <= 30:
            # Acceptable equilibrium
            return 0.6
        elif gen_range <= 50:
            # Poor equilibrium
            return 0.3
        else:
            # Not at equilibrium - levels are too different
            return 0.1
    
    def _detect_liquid_levels(self, frame: np.ndarray, n_vessels: int = None) -> List[int]:
        """Detect liquid levels in vessels using pixel color detection.
        
        Detect vessels by finding columns with significant colored (saturated) pixels.
        Then detect the top y-coordinate of liquid in each vessel.
        Only consider the main liquid region (bottom half of frame typically).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Detect any colored (saturated) liquid
        saturation = hsv[:, :, 1]
        liquid_mask = saturation > 80  # Higher threshold to avoid compression artifacts
        
        # Only consider the main region (exclude top 1/3 which may have labels)
        main_region_start = h // 3
        main_mask = np.zeros_like(liquid_mask)
        main_mask[main_region_start:, :] = liquid_mask[main_region_start:, :]
        
        # Find columns with significant liquid
        col_sums = np.sum(main_mask, axis=0)
        
        # Find peaks (vessel centers) by finding regions with high column sums
        # Use a higher threshold to filter out noise
        threshold = np.max(col_sums) * 0.5 if np.max(col_sums) > 0 else 0
        
        vessel_regions = []
        in_vessel = False
        start_col = 0
        
        for x in range(w):
            if col_sums[x] > threshold and not in_vessel:
                in_vessel = True
                start_col = x
            elif col_sums[x] <= threshold and in_vessel:
                in_vessel = False
                # Only add if region is wide enough (actual vessel, not noise)
                region_width = x - start_col
                if region_width > w // 20:  # At least 5% of width
                    center = (start_col + x) // 2
                    vessel_regions.append((center, region_width))
        
        if in_vessel:
            region_width = w - start_col
            if region_width > w // 20:
                vessel_regions.append(((start_col + w) // 2, region_width))
        
        # Sort by x position and take the main vessels
        vessel_regions.sort(key=lambda x: x[0])
        vessel_cols = [v[0] for v in vessel_regions]
        
        # If no vessels found, fall back to equal division
        if len(vessel_cols) < 2:
            n_vessels = n_vessels or 3
            vessel_cols = [(i * 2 + 1) * w // (n_vessels * 2) for i in range(n_vessels)]
        
        # Detect liquid levels at each vessel column
        levels = []
        for x in vessel_cols:
            # Look for top of liquid in a small region around x
            x1 = max(0, x - 30)
            x2 = min(w, x + 30)
            col_mask = main_mask[:, x1:x2]
            row_sums = np.sum(col_mask, axis=1)
            liquid_rows = np.where(row_sums > 10)[0]
            if len(liquid_rows) > 0:
                levels.append(liquid_rows[0])
        
        return levels
    
    
    def _evaluate_flow_process(self, frames: List[np.ndarray]) -> float:
        """Evaluate if flow process is realistic."""
        if len(frames) < 5:
            return 0.5
        
        # Track level variance over time
        variances = []
        for frame in frames[::max(1, len(frames)//10)]:
            levels = self._detect_liquid_levels(frame)
            if len(levels) >= 2:
                variances.append(np.std(levels))
        
        if len(variances) < 3:
            return 0.5
        
        # Variance should decrease over time (approaching equilibrium)
        decreasing = 0
        for i in range(1, len(variances)):
            if variances[i] <= variances[i-1] + 5:  # Allow small fluctuations
                decreasing += 1
        
        return decreasing / (len(variances) - 1)
    
    def _evaluate_volume_conservation(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if total liquid volume is conserved."""
        def count_liquid_pixels(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            return np.sum(saturation > 50)
        
        first_volume = count_liquid_pixels(first_frame)
        final_volume = count_liquid_pixels(gen_final)
        
        if first_volume == 0:
            return 0.5
        
        ratio = final_volume / first_volume
        
        if 0.9 <= ratio <= 1.1:
            return 1.0
        elif 0.8 <= ratio <= 1.2:
            return 0.7
        elif 0.6 <= ratio <= 1.4:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_visual_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if vessel structure is preserved."""
        # Detect vessel outlines (dark lines)
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge density
        gen_edge_density = np.sum(gen_edges > 0) / gen_edges.size
        gt_edge_density = np.sum(gt_edges > 0) / gt_edges.size
        
        if gt_edge_density > 0:
            ratio = gen_edge_density / gt_edge_density
            return min(1.0, max(0.3, 1.0 - abs(1 - ratio)))
        
        return 0.5


# Export mapping for this batch
OPEN60_EVALUATORS_PART6 = {
    'O-36_grid_shift_data-generator': GridShiftEvaluator,
    'O-37_light_sequence_data-generator': LightSequenceEvaluator,
    'O-38_majority_color_data-generator': MajorityColorEvaluator,
    'O-44_rotation_puzzle_data-generator': RotationPuzzleEvaluator,
    'O-45_sequence_completion_data-generator': SequenceCompletionEvaluator,
    'O-47_sliding_puzzle_data-generator': SlidingPuzzleEvaluator,
    'O-52_traffic_light_data-generator': TrafficLightEvaluator,
    'O-53_clock_data-generator': ClockTimeEvaluator,
    'O-55_rotation_data-generator': RotationEvaluator,
    'O-75_communicating_vessels_data-generator': CommunicatingVesselsEvaluator,
