"""Evaluator for G-31_directed_graph_navigation_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-31_directed_graph_navigation_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class DirectedGraphNavigationEvaluator(BaseEvaluator):
    """
    G-31: Directed graph navigation evaluator.
    
    CRITICAL RULES:
    1. Blue triangle (agent) must move from green circle to red circle
    2. Agent must reach the red circle (endpoint)
    3. All circle colors (green, red) must NOT change
    """
    
    TASK_WEIGHTS = {
        'completion': 0.25,           # Agent reaches red endpoint
        'circles_preserved': 0.20,    # Circle colors unchanged
        'path_quality': 0.05,         # Smooth movement (legacy)
        'direction_compliance': 0.35, # Follows arrows (arrow-based detection)
        'movement_legality': 0.10,    # Follows edges
        'graph_fidelity': 0.05        # Graph structure preserved
    }
    
    def _count_circle_colors(self, frame: np.ndarray) -> Tuple[int, int]:
        """Count green and red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_count = np.sum(green_mask > 0)
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_count = np.sum(red_mask > 0)
        
        return green_count, red_count
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate directed graph navigation task.
        
        CRITICAL RULES:
        1. Agent must reach red endpoint
        2. Circle colors (green, red) must NOT change significantly
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # CRITICAL: Check if circle colors are preserved
        first_green, first_red = self._count_circle_colors(first_frame)
        final_green, final_red = self._count_circle_colors(last_frame)
        
        first_total = first_green + first_red
        final_total = final_green + final_red
        
        # Circle colors should not change dramatically
        total_change = abs(final_total - first_total) / max(first_total, 1)
        
        if total_change > 1.0:  # More than 100% increase
            # Circle colors changed significantly - task failed
            scores['circles_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['path_quality'] = 0.0
            scores['direction_compliance'] = 0.0
            scores['movement_legality'] = 0.0
            scores['graph_fidelity'] = 0.0
            self._last_task_details = scores
            self._last_task_details['circles_changed'] = True
            return 0.0
        else:
            scores['circles_preserved'] = max(0, 1.0 - total_change)
        
        # Detect nodes and agent
        nodes_first = self._detect_nodes(first_frame)
        gen_agent_final = self._detect_agent(last_frame)
        
        # 1. Completion: Check if agent reached red endpoint
        if gen_agent_final is not None and nodes_first.get('end') is not None:
            end_pos = nodes_first['end']
            dist = np.sqrt((gen_agent_final[0] - end_pos[0])**2 + 
                          (gen_agent_final[1] - end_pos[1])**2)
            if dist < 50:
                scores['completion'] = 1.0
            elif dist < 100:
                scores['completion'] = 0.3  # STRICT: Not at endpoint
            else:
                scores['completion'] = 0.0  # STRICT: Failed to reach endpoint
        else:
            scores['completion'] = 0.0
        
        # 3. Path quality (legacy smooth movement check)
        agent_positions = self._track_agent(video_frames)
        if len(agent_positions) >= 2:
            # Check for smooth movement (no teleporting)
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 150 or dy > 150:
                    large_jumps += 1
            scores['path_quality'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['path_quality'] = 0.0

        # 4. Direction compliance: Check if agent follows arrow directions
        # Uses arrow detection from gt_first_frame
        scores['direction_compliance'] = self._evaluate_direction_compliance_arrow(
            agent_positions, nodes_first, gt_first_frame
        )

        # 5. Movement legality: Check if agent moves along edges
        scores['movement_legality'] = self._evaluate_movement_legality(agent_positions, nodes_first)

        # 6. Graph fidelity: Check if graph structure is preserved
        scores['graph_fidelity'] = self._evaluate_graph_fidelity(first_frame, last_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect blue triangular agent position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest blue region (the agent)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _track_agent(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track agent position across all frames."""
        positions = []
        for frame in frames:
            pos = self._detect_agent(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_nodes(self, frame: np.ndarray) -> Dict:
        """Detect graph nodes (green=start, red=end, white=intermediate)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        nodes = {'start': None, 'end': None, 'intermediate': []}
        
        # Green (start node)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['start'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        # Red (end node)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['end'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        return nodes
    
    def _evaluate_path_length(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent took shortest path."""
        if not agent_positions or not nodes.get('end'):
            return 0.0
        
        # Check if agent reached end node
        final_pos = agent_positions[-1]
        end_node = nodes['end']
        
        dist_to_end = np.sqrt((final_pos[0] - end_node[0])**2 + (final_pos[1] - end_node[1])**2)
        
        # If agent reached end (within threshold)
        if dist_to_end < 50:
            # Count number of significant position changes (steps)
            steps = 0
            prev_pos = agent_positions[0]
            for pos in agent_positions[1:]:
                dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                if dist > 20:  # Significant movement
                    steps += 1
                    prev_pos = pos
            
            # Fewer steps is better (assuming optimal is 3-5 steps for typical graph)
            if steps <= 5:
                return 1.0
            elif steps <= 7:
                return 0.8
            elif steps <= 10:
                return 0.6
            else:
                return 0.4
        else:
            # Didn't reach end
            return max(0.2, 1.0 - dist_to_end / 500)
    
    def _evaluate_direction_compliance_arrow(
        self,
        agent_positions: List[Tuple[int, int]],
        nodes: Dict,
        gt_first_frame: Optional[np.ndarray]
    ) -> float:
        """Evaluate direction compliance using actual arrow detection from GT first frame.
        
        Detects arrows in the graph, builds directed edge set, then checks
        if the agent's path follows those directed edges.
        """
        if len(agent_positions) < 2:
            return 0.5
        if gt_first_frame is None:
            # Fallback to old proxy method
            return self._evaluate_direction_compliance(agent_positions, nodes)

        # Step 1: Detect all graph nodes (circles) from GT first frame
        circle_centers = self._detect_all_nodes(gt_first_frame)
        if len(circle_centers) < 2:
            return 0.5

        # Step 2: Detect directed edges (arrows) from GT first frame
        directed_edges = self._detect_arrows(gt_first_frame, circle_centers)
        if not directed_edges:
            # No arrows detected, fall back to old method
            return self._evaluate_direction_compliance(agent_positions, nodes)

        # Step 3: Map agent positions to nearest nodes
        def nearest_node(pos):
            best = None
            best_dist = float('inf')
            for i, c in enumerate(circle_centers):
                d = np.sqrt((pos[0] - c[0])**2 + (pos[1] - c[1])**2)
                if d < best_dist:
                    best_dist = d
                    best = i
            return best, best_dist

        # Step 4: Build agent node sequence (filter stationary frames)
        node_sequence = []
        prev_node = None
        for pos in agent_positions:
            node_idx, dist = nearest_node(pos)
            if node_idx != prev_node:
                node_sequence.append(node_idx)
                prev_node = node_idx

        if len(node_sequence) < 2:
            return 0.5

        # Step 5: Check each transition against directed edges
        legal_moves = 0
        total_moves = 0
        for i in range(1, len(node_sequence)):
            src = node_sequence[i - 1]
            dst = node_sequence[i]
            if src != dst:
                total_moves += 1
                if (src, dst) in directed_edges:
                    legal_moves += 1

        if total_moves == 0:
            return 0.5

        return legal_moves / total_moves

    def _detect_all_nodes(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect all circle nodes in the graph (white circles with black border)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=10,
            maxRadius=60
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return [(int(c[0]), int(c[1])) for c in circles]

        # Fallback: use contour detection
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 20000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.5:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centers.append((cx, cy))
        return centers

    def _detect_arrows(
        self,
        frame: np.ndarray,
        node_centers: List[Tuple[int, int]]
    ) -> set:
        """Detect directed edges (arrows) in the graph.
        
        Returns a set of (src_node_idx, dst_node_idx) pairs.
        
        Strategy:
        1. Threshold to get dark pixels (arrows + circle outlines).
        2. Mask out circles to isolate arrow lines.
        3. Detect line segments via HoughLinesP.
        4. Merge collinear short segments into full arrows.
        5. Determine arrow direction by detecting V-shape (arrowhead)
           near each endpoint using contour shape analysis.
        6. Map endpoints to nearest nodes to build directed edge set.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold: keep dark pixels (black arrows on white background)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Create a mask that removes circle areas
        circle_mask = np.zeros_like(binary)
        for cx, cy in node_centers:
            cv2.circle(circle_mask, (cx, cy), 30, 255, -1)
        arrow_binary = cv2.bitwise_and(binary, cv2.bitwise_not(circle_mask))

        # Detect line segments
        lines = cv2.HoughLinesP(
            arrow_binary,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=15,
            maxLineGap=15
        )
        if lines is None:
            return set()

        # --- Step 1: Merge nearly-collinear segments into full arrows ---
        def angle_of(x1, y1, x2, y2):
            """Angle in degrees of segment, normalized to [0, 180)."""
            a = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            return a

        def endpoints_close(p1, p2, thresh=20):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < thresh

        def merge_segments(segs, angle_thresh=15, dist_thresh=20):
            """Greedy merge of segments with similar angle and close endpoints."""
            merged = []
            used = [False] * len(segs)
            for i, s1 in enumerate(segs):
                if used[i]:
                    continue
                x1, y1, x2, y2 = s1
                for j, s2 in enumerate(segs):
                    if i == j or used[j]:
                        continue
                    x3, y3, x4, y4 = s2
                    if abs(angle_of(x1,y1,x2,y2) - angle_of(x3,y3,x4,y4)) > angle_thresh:
                        continue
                    # Check if any endpoints are close
                    pts1 = [(x1,y1),(x2,y2)]
                    pts2 = [(x3,y3),(x4,y4)]
                    close = any(endpoints_close(a,b,dist_thresh) for a in pts1 for b in pts2)
                    if close:
                        # Merge: take the two farthest endpoints
                        all_pts = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                        max_d = 0
                        best = (x1,y1,x2,y2)
                        for pi in all_pts:
                            for pj in all_pts:
                                d = np.sqrt((pi[0]-pj[0])**2+(pi[1]-pj[1])**2)
                                if d > max_d:
                                    max_d = d
                                    best = (pi[0],pi[1],pj[0],pj[1])
                        x1,y1,x2,y2 = best
                        used[j] = True
                used[i] = True
                merged.append((x1,y1,x2,y2))
            return merged

        raw_segs = [tuple(l[0]) for l in lines]
        merged_segs = merge_segments(raw_segs)

        # --- Step 2: Determine arrowhead end using local contour analysis ---
        def has_arrowhead(pt, binary_img, line_angle_deg, radius=12):
            """
            Check if there is an arrowhead (V-shape triangle) near pt.
            
            Strategy: Extract a small ROI around pt, find contours,
            check if any contour is roughly triangular (3-4 vertices
            after polygon approximation) and is pointing toward pt.
            """
            h, w = binary_img.shape
            x, y = int(pt[0]), int(pt[1])
            x1c = max(0, x - radius)
            x2c = min(w, x + radius)
            y1c = max(0, y - radius)
            y2c = min(h, y + radius)
            roi = binary_img[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                return False, 0

            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 5:
                    continue
                # Approximate polygon
                epsilon = 0.04 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                # Triangle = 3 vertices, arrow head is roughly triangular
                if 3 <= len(approx) <= 5:
                    return True, area
            return False, 0

        def nearest_node_idx(pt):
            best = None
            best_dist = float('inf')
            for i, c in enumerate(node_centers):
                d = np.sqrt((pt[0]-c[0])**2+(pt[1]-c[1])**2)
                if d < best_dist:
                    best_dist = d
                    best = i
            return best, best_dist

        directed_edges = set()

        for seg in merged_segs:
            x1, y1, x2, y2 = seg
            pt1 = (x1, y1)
            pt2 = (x2, y2)

            node1, dist1 = nearest_node_idx(pt1)
            node2, dist2 = nearest_node_idx(pt2)

            if node1 is None or node2 is None or node1 == node2:
                continue
            # At least one end must be near a node
            if dist1 > 70 and dist2 > 70:
                continue

            line_angle = angle_of(x1, y1, x2, y2)

            # Check for arrowhead near each endpoint
            head1, area1 = has_arrowhead(pt1, arrow_binary, line_angle)
            head2, area2 = has_arrowhead(pt2, arrow_binary, line_angle)

            if head2 and not head1:
                # pt2 is arrowhead -> arrow points TO node2 -> edge: node1 -> node2
                if dist1 <= 70:
                    directed_edges.add((node1, node2))
            elif head1 and not head2:
                # pt1 is arrowhead -> arrow points TO node1 -> edge: node2 -> node1
                if dist2 <= 70:
                    directed_edges.add((node2, node1))
            else:
                # Both or neither detected - fallback to pixel density
                def dark_pixel_density(pt, img, radius=8):
                    h, w = img.shape
                    x, y = int(pt[0]), int(pt[1])
                    x1c = max(0, x - radius)
                    x2c = min(w, x + radius)
                    y1c = max(0, y - radius)
                    y2c = min(h, y + radius)
                    region = img[y1c:y2c, x1c:x2c]
                    if region.size == 0:
                        return 0
                    return np.sum(region > 0) / region.size

                d1 = dark_pixel_density(pt1, arrow_binary)
                d2 = dark_pixel_density(pt2, arrow_binary)
                if d2 >= d1 and dist1 <= 70:
                    directed_edges.add((node1, node2))
                elif d1 > d2 and dist2 <= 70:
                    directed_edges.add((node2, node1))

        return directed_edges

    def _evaluate_direction_compliance(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Fallback: Evaluate if agent follows arrow directions (proxy method)."""
        if len(agent_positions) < 2:
            return 0.5
        
        # For now, check if movement is generally forward (left to right or top to bottom)
        # This is a simplification; full implementation would need edge detection
        forward_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dx = agent_positions[i][0] - agent_positions[i-1][0]
            dy = agent_positions[i][1] - agent_positions[i-1][1]
            
            if abs(dx) > 10 or abs(dy) > 10:  # Significant movement
                total_moves += 1
                # Generally forward progress (towards end)
                if nodes.get('end') and nodes.get('start'):
                    # Check if moving towards end
                    prev_dist = np.sqrt((agent_positions[i-1][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i-1][1] - nodes['end'][1])**2)
                    curr_dist = np.sqrt((agent_positions[i][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i][1] - nodes['end'][1])**2)
                    if curr_dist < prev_dist:
                        forward_moves += 1
        
        return forward_moves / max(1, total_moves)
    
    def _evaluate_movement_legality(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent moves along edges (not jumping)."""
        if len(agent_positions) < 2:
            return 0.5
        
        # Check for smooth movement (no large jumps)
        smooth_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dist = np.sqrt((agent_positions[i][0] - agent_positions[i-1][0])**2 + 
                          (agent_positions[i][1] - agent_positions[i-1][1])**2)
            
            if dist > 5:  # Significant movement
                total_moves += 1
                if dist < 100:  # Reasonable step size
                    smooth_moves += 1
        
        return smooth_moves / max(1, total_moves)
    
    def _evaluate_graph_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if graph structure is preserved."""
        # Compare edge structures using edge detection
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge maps
        intersection = np.sum((gen_edges > 0) & (gt_edges > 0))
        union = np.sum((gen_edges > 0) | (gt_edges > 0))
        
        if union > 0:
            return intersection / union
        return 0.5
