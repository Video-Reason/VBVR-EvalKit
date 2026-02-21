"""Evaluator for O-30_bookshelf_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-30_bookshelf_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class BookshelfEvaluator(BaseEvaluator):
    """
    O-30: Bookshelf (Height Clustering)
    
    Task: Insert new books into correct positions based on height clustering.
    Books go to the cluster with closest average height.
    
    Rule-based evaluation:
    1. Cluster identification (30%) - Correct cluster boundaries
    2. Representative height calculation (25%) - Correct averages
    3. Matching and insertion (30%) - Books at correct positions
    4. Sorting constraint (15%) - Multiple books sorted by height
    """
    
    TASK_WEIGHTS = {
        'cluster_identification': 0.30,
        'height_calculation': 0.25,
        'matching_insertion': 0.30,
        'sorting': 0.15
    }
    
    def _detect_book_heights(self, frame: np.ndarray) -> List[int]:
        """Detect vertical book heights using color-based detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        heights = []
        
        # Method 1: Color-based detection (gold and azure books)
        if hsv is not None:
            # Gold: H ~15-45, high S
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Azure/Blue: H ~90-130, high S
            lower_azure = np.array([90, 50, 50])
            upper_azure = np.array([130, 255, 255])
            azure_mask = cv2.inRange(hsv, lower_azure, upper_azure)
            
            # Combine color masks
            color_mask = gold_mask | azure_mask
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w * 0.8 and h > 20:  # Vertical book (allow some tolerance)
                    heights.append(h)
        
        # Method 2: Grayscale threshold fallback (try multiple thresholds)
        if len(heights) < 2:
            for thresh_val in [100, 150, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                temp_heights = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > w and h > 20:  # Vertical book
                        temp_heights.append(h)
                
                if len(temp_heights) > len(heights):
                    heights = temp_heights
        
        return sorted(heights)
    
    def _analyze_book_arrangement(self, frame: np.ndarray) -> Dict:
        """Analyze book arrangement using color-based detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        books = []
        
        # Method 1: Color-based detection (gold and azure books)
        if hsv is not None:
            # Gold: H ~15-45, high S
            lower_gold = np.array([15, 50, 50])
            upper_gold = np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Azure/Blue: H ~90-130, high S
            lower_azure = np.array([90, 50, 50])
            upper_azure = np.array([130, 255, 255])
            azure_mask = cv2.inRange(hsv, lower_azure, upper_azure)
            
            # Combine color masks
            color_mask = gold_mask | azure_mask
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Filter: vertical shape (height > width), minimum height, minimum area
                if h > w * 0.8 and h > 20 and area > 500:
                    books.append({'x': x, 'height': h, 'y': y, 'width': w})
        
        # Method 2: Grayscale threshold fallback (try multiple thresholds)
        if len(books) < 2:
            for thresh_val in [100, 150, 200]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                temp_books = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    if h > w and h > 20 and area > 500:
                        temp_books.append({'x': x, 'height': h, 'y': y, 'width': w})
                
                if len(temp_books) > len(books):
                    books = temp_books
        
        # Sort by x-position
        books.sort(key=lambda b: b['x'])
        
        return {
            'book_count': len(books),
            'heights': [b['height'] for b in books],
            'positions': [b['x'] for b in books]
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate bookshelf insertion accuracy - STRICT GT comparison."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        # STRICT: Compare directly with GT final frame
        # The task requires books to be inserted in correct positions based on height clustering
        final_diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
        
        # 1. Cluster identification (30%): STRICT comparison with GT
        if final_diff < 10:
            scores['cluster_identification'] = 1.0
        elif final_diff < 25:
            scores['cluster_identification'] = 0.3
        else:
            scores['cluster_identification'] = 0.0
        
        # 2. Height calculation (25%): STRICT comparison with GT
        if final_diff < 10:
            scores['height_calculation'] = 1.0
        elif final_diff < 25:
            scores['height_calculation'] = 0.3
        else:
            scores['height_calculation'] = 0.0
        
        # 3. Matching insertion (30%): STRICT comparison with GT
        if final_diff < 10:
            scores['matching_insertion'] = 1.0
        elif final_diff < 25:
            scores['matching_insertion'] = 0.3
        else:
            scores['matching_insertion'] = 0.0
        
        # 4. Sorting (15%): STRICT comparison with GT
        if final_diff < 10:
            scores['sorting'] = 1.0
        elif final_diff < 25:
            scores['sorting'] = 0.3
        else:
            scores['sorting'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_task_specific_old(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """OLD: Evaluate bookshelf insertion accuracy using detection."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        # Analyze arrangements
        gen_arr = self._analyze_book_arrangement(gen_final)
        gt_arr = self._analyze_book_arrangement(gt_final)
        
        # 1. Cluster identification (30%): Compare book counts
        # Rule: Correctly identify height-based clusters using eps threshold
        if gt_arr['book_count'] > 0:
            count_ratio = min(gen_arr['book_count'], gt_arr['book_count']) / max(gen_arr['book_count'], gt_arr['book_count'])
            # Perfect match gets full score
            if count_ratio == 1.0:
                scores['cluster_identification'] = 1.0
            elif count_ratio >= 0.9:
                scores['cluster_identification'] = 0.9
            else:
                scores['cluster_identification'] = max(0.3, count_ratio)
        else:
            # No GT books - check if generated also has no books
            scores['cluster_identification'] = 1.0 if gen_arr['book_count'] == 0 else 0.3
        
        # 2. Height calculation (25%): Compare height distributions
        # Rule: Representative height = average of cluster heights
        if gen_arr['heights'] and gt_arr['heights']:
            gen_mean_h = np.mean(gen_arr['heights'])
            gt_mean_h = np.mean(gt_arr['heights'])
            
            if gt_mean_h > 0:
                height_ratio = min(gen_mean_h, gt_mean_h) / max(gen_mean_h, gt_mean_h)
                # Rule: calculation precision should be within 5% for full score
                if height_ratio >= 0.95:
                    scores['height_calculation'] = 1.0
                elif height_ratio >= 0.90:
                    scores['height_calculation'] = 0.8
                else:
                    scores['height_calculation'] = max(0.3, height_ratio)
            else:
                scores['height_calculation'] = 0.2  # Detection failed
        else:
            # No heights detected
            scores['height_calculation'] = 0.5 if not gt_arr['heights'] else 0.0
        
        # 3. Matching insertion (30%): Compare book positions
        # Rule: Each new book should be at the end of its matched cluster
        if gen_arr['positions'] and gt_arr['positions']:
            # Compare position sequences
            matched_positions = 0
            for gen_pos in gen_arr['positions']:
                for gt_pos in gt_arr['positions']:
                    if abs(gen_pos - gt_pos) < 30:  # Within 30 pixels
                        matched_positions += 1
                        break
            position_match = matched_positions / max(len(gt_arr['positions']), 1)
            scores['matching_insertion'] = position_match
        else:
            scores['matching_insertion'] = 0.5 if not gt_arr['positions'] else 0.0
        
        # 4. Sorting (15%): Check if heights are sorted within clusters
        if len(gen_arr['heights']) >= 2:
            # Check for local sorting (within clusters)
            sorted_count = 0
            for i in range(1, len(gen_arr['heights'])):
                if gen_arr['heights'][i] >= gen_arr['heights'][i-1] * 0.8:
                    sorted_count += 1
            scores['sorting'] = sorted_count / (len(gen_arr['heights']) - 1)
        else:
            scores['sorting'] = 0.8  # Single book is trivially sorted
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
