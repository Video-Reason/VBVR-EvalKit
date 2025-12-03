"""
Subway Pathfinding Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/PathFinding/create_subway.py

Minimal modifications to fit VMEvalKit interface.
All generation logic is preserved from Tin's original implementation.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import random
import json
from matplotlib import cm
from matplotlib import colormaps
import matplotlib.colors as mcolors
from PIL import Image
import itertools
import copy
from collections import defaultdict
import os
import tempfile
from typing import Dict, Any, Optional, Sequence

# ============================================
# Tin's Original Constants and Functions (UNCHANGED)
# ============================================

# assuming your canvas is 18x18

valid_moves = {'A':[(0, -1), (-1, 0), (1, 0)],
               'B':[(0, -1), (0, 1), (-1, 0)],
               'C':[(0, 1), (-1, 0), (1, 0)],
               'D':[(0, -1), (0, 1), (1, 0)]}

starting_moves = {'A':(0, -1),
                  'B':(-1, 0),
                  'C':(0, 1),
                  'D':(1, 0)}

stations_points = {'A': [(8, 16), (9, 16), (10, 16)],
                'B': [(16, 8), (16, 9), (16, 10)],
                'C': [(8, 2), (9, 2), (10, 2)],
                'D': [(2, 8), (2, 9), (2, 10)]}

def get_colors_from_colormap(colormap_name, num_colors):
    colormap = colormaps[colormap_name]
    indicies = np.arange(num_colors)
    colors = [colormap(i) for i in indicies]
    return colors

def rgba_to_color_name(rgba):
    colors = mcolors.CSS4_COLORS
    
    input_rgb = rgba[:3]

    closest_color = min(colors, key=lambda name: np.linalg.norm(np.array(mcolors.to_rgba(colors[name])[:3]) - np.array(input_rgb)))
    
    return closest_color

def draw_lines(all_routes, thickness, node_to_label, color, add_agent=False, agent_destination=None, agent_path_idx=None):
    fig, ax = plt.subplots(figsize=(18, 18), dpi=500)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 18)
    ax.axis('off')

    ax.text(8.45, 16.5, 'A', fontsize=100, color='k', fontweight='bold')
    ax.text(16.5, 8.6, 'B', fontsize=100, color='k', fontweight='bold')
    ax.text(8.45, 0.6, 'C', fontsize=100, color='k', fontweight='bold')
    ax.text(0.4, 8.6, 'D', fontsize=100, color='k', fontweight='bold')

    connections = []
    for i, paths in enumerate(all_routes):
        connection = dict()
        for path in paths['path']:
            x1, y1 = path[0][0], path[0][1]
            x2, y2 = path[1][0], path[1][1]
            ax.plot([x1, x2], [y1, y2], color=color[i], linestyle='solid', linewidth=thickness)

        connection['start'] = node_to_label[str(paths['path'][0][0])]
        connection['end'] = node_to_label[str(paths['path'][-1][-1])]
        connection['color'] = rgba_to_color_name(color[i])
        connections.append(connection)

    # Add agent icon at destination if requested (for last frame)
    if add_agent and agent_destination is not None and agent_path_idx is not None:
        # Get the destination coordinates
        dest_x, dest_y = agent_destination
        # Draw agent as a larger circle with distinct color
        agent_circle = plt.Circle((dest_x, dest_y), 0.5, color='red', zorder=100, linewidth=3, edgecolor='darkred')
        ax.add_artist(agent_circle)
        # Add a marker inside
        ax.plot(dest_x, dest_y, marker='*', markersize=30, color='yellow', zorder=101)

    return fig, connections
    
def convert_fig_to_pil(fig):
    fig.canvas.draw()
    
    # Use buffer_rgba() for matplotlib 3.8+ compatibility
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # Calculate actual dimensions from buffer size (handles high-DPI displays)
    # Buffer is RGBA, so divide by 4 to get number of pixels
    num_pixels = len(buf) // 4
    # Assume square image, calculate side length
    side_length = int(np.sqrt(num_pixels))
    
    # Reshape using calculated dimensions
    buf = buf.reshape(side_length, side_length, 4)
    
    # Convert RGBA to RGB by dropping alpha channel
    image = Image.fromarray(buf[:, :, :3])

    return image

def generate_subway_routes():
    """Generate subway routes using Tin's original algorithm."""
    node_to_label = dict()

    for k, v in stations_points.items():
        for i in range(len(v)):
            node_to_label[str(v[i])] = k

    images = []
    stations = ['A', 'B', 'C', 'D']
    random.shuffle(stations)

    path_nums = [1, 2, 3]

    while True:
        cnt = 0
        while cnt < len(path_nums):
            p = path_nums[cnt]
            path_counter = defaultdict(int)
            visited = []
            all_routes = []
            cross_dest = []

            for station in stations:
                path_counter[station] += 0

            for station in stations:
                start_list = copy.deepcopy(stations_points[station])
                random.shuffle(start_list)

                possible_destinations = []

                for k, v in stations_points.items():
                    if k == station:
                        continue

                    if path_counter[k] == p:
                        continue
                    
                    for i in range(len(v)):
                        if v[i] not in cross_dest:
                            possible_destinations.append(v[i])

                for i in list(np.arange(1, p+1)):
                    if path_counter[station] == p:
                        break
                    
                    start = start_list.pop()
                    root = start
                    end = (start[0] + starting_moves[station][0], start[1] + starting_moves[station][1])

                    routes = dict()
                    routes['path'] = []
                    
                    while True:
                        if end in possible_destinations:
                            break

                        if [start, end] not in visited and [end, start] not in visited and end[0] > 2 and end[0] < 16 and end[1] > 2 and end[1] < 16:
                            routes['path'].append([start, end])
                            visited.append([start, end])
                            temp_moves = copy.deepcopy(valid_moves[station])

                        else:
                            if len(routes['path']) < 1:
                                break
                            old_data = routes['path'][-1]
                            start, end = old_data[0], old_data[1]

                        if len(temp_moves) < 1:
                            if len(routes['path']) > 1:
                                routes['path'].pop(-1)
                            else:
                                break
                            old_data = routes['path'][-1]
                            start, end = old_data[0], old_data[1]
                            temp_moves = copy.deepcopy(valid_moves[station])
                        
                        route = random.choice(temp_moves)
                        temp_moves.pop(temp_moves.index(route))
                        start = end
                        end = (start[0] + route[0], start[1] + route[1])

                    if end in possible_destinations and [start, end] not in visited and [end, start] not in visited:
                        routes['path'].append([start, end])
                        visited.append([start, end])
                        all_routes.append(routes)
                        path_counter[station] += 1
                        cross_dest.append(end)
                        cross_dest.append(root)

                        for k, v in stations_points.items():
                            if end in v:
                                path_counter[k] += 1

            sw = 1
            for station in stations:
                if path_counter[station] == p: 
                    sw *= 1
                else:
                    sw *= 0

            if sw == 1:              
                images.append(all_routes)
                cnt += 1

        if len(images) == 45:
            break
    
    return images, node_to_label

# ============================================
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = 10, difficulties: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Generate subway pathfinding dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate
        difficulties: List of difficulty levels to generate.
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    temp_dir = "subway_pathfinding"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Setup difficulties
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]
    
    # ============================================
    # Tin's Original Generation Logic (UNCHANGED)
    # ============================================
    
    # Generate subway routes (this generates 45 variations)
    images, node_to_label = generate_subway_routes()
    
    # Define difficulty levels based on visual complexity
    # Easy: larger images, thicker lines (easier to see)
    # Medium: smaller images or thinner lines
    # Hard: smaller images with thinner lines (harder to trace)
    
    difficulty_configs = {
        'easy': {'image_sizes': [1024], 'thicknesses': [20]},
        'medium': {'image_sizes': [1024, 512], 'thicknesses': [10, 20]},
        'hard': {'image_sizes': [512], 'thicknesses': [10]}
    }
    
    # Build available configurations based on requested difficulties
    available_configs = []
    for diff in diffs:
        if diff in difficulty_configs:
            config = difficulty_configs[diff]
            for img_size in config['image_sizes']:
                for thickness in config['thicknesses']:
                    available_configs.append({
                        'difficulty': diff,
                        'image_size': img_size,
                        'thickness': thickness
                    })
    
    # If no configs available, return empty
    if not available_configs:
        return {
            "name": "subway_pathfinding_tasks",
            "pairs": [],
            "source": "tin_tasks",
            "total_samples": 0,
            "difficulties": list(diffs)
        }
    
    test_samples = []

    # Generate num_samples by randomly selecting from available configurations
    for sample_idx in range(num_samples):
        # Randomly select a route configuration
        all_routes = random.choice(images)
        color = get_colors_from_colormap('tab10', len(all_routes))
        
        # Randomly select difficulty configuration
        config = random.choice(available_configs)
        image_size = config['image_size']
        thickness = config['thickness']
        difficulty = config['difficulty']
        
        # Pick a random route to focus on for this sample
        focus_route_idx = random.randint(0, len(all_routes) - 1)
        focus_route = all_routes[focus_route_idx]
        
        # Get source and destination
        source_station = node_to_label[str(focus_route['path'][0][0])]
        dest_station = node_to_label[str(focus_route['path'][-1][-1])]
        dest_coords = focus_route['path'][-1][-1]
        
        # Generate first frame (without agent)
        fig, connections = draw_lines(all_routes, thickness, node_to_label, color, 
                                     add_agent=False)
        plt.tight_layout(pad=0.0)
        image = convert_fig_to_pil(fig)
        plt.close(fig)
        image = image.resize((image_size, image_size))
        
        first_frame_name = f"{sample_idx + 1}_first.png"
        image.save(os.path.join(temp_dir, first_frame_name))
        
        # Generate last frame (with agent at destination)
        fig, connections = draw_lines(all_routes, thickness, node_to_label, color,
                                     add_agent=True, agent_destination=dest_coords,
                                     agent_path_idx=focus_route_idx)
        plt.tight_layout(pad=0.0)
        image = convert_fig_to_pil(fig)
        plt.close(fig)
        image = image.resize((image_size, image_size))
        
        last_frame_name = f"{sample_idx + 1}_last.png"
        image.save(os.path.join(temp_dir, last_frame_name))
       
        # Calculate path length for metadata
        path_length = len(focus_route['path'])
        
        # Get the path color name
        path_color = rgba_to_color_name(color[focus_route_idx])
        
        # Tin's original data structure + minimal VMEvalKit fields
        test_sample = {
            "sample_id": f"sample_{sample_idx + 1:04d}",
            "prompt": f"Create a video to show how an agent travels from station {source_station} to station {dest_station} by following the {path_color} path/line on the subway map.",
            "first_frame": first_frame_name,
            "last_frame": last_frame_name,
            "source_station": source_station,
            "destination_station": dest_station,
            "path_color": path_color,
            "difficulty": difficulty,
            "metadata": {
                "linewidth": thickness,
                "image_size": image_size,
                "all_connections": connections,
                "focus_route_idx": focus_route_idx,
                "destination_coords": dest_coords,
                "path_length": path_length,
                "num_routes": len(all_routes)
            },
            # VMEvalKit required fields
            "id": f"subway_pathfinding_{sample_idx:04d}",
            "domain": "subway_pathfinding",
            "first_image_path": os.path.join(temp_dir, first_frame_name),
            "final_image_path": os.path.join(temp_dir, last_frame_name),
        }
        test_samples.append(test_sample)

    return {
        "name": "subway_pathfinding_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples),
        "difficulties": list(diffs)
    }


if __name__ == "__main__":
    dataset = create_dataset(num_samples=10, difficulties=['easy', 'medium', 'hard'])
    print(f"Generated {dataset['total_samples']} samples")
    print(f"Difficulties: {dataset['difficulties']}")
    print(f"Sample preview: {[s['sample_id'] for s in dataset['pairs'][:3]]}")
