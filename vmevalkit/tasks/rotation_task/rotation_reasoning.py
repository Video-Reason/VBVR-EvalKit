#!/usr/bin/env python3
"""
3D Mental Rotation Task for VMEvalKit

This module generates 3D voxel structures (snake-like configurations) and creates
mental rotation tasks where models must demonstrate spatial reasoning by showing
how these structures appear when rotated to different viewpoints.

The task evaluates a model's ability to:
1. Understand 3D spatial relationships from 2D projections
2. Mentally rotate objects in 3D space
3. Generate videos showing smooth rotation transitions
4. Accurately predict the final appearance after rotation

Author: VMEvalKit Team
"""

import math
import os
import json
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Set, Iterable, Optional
from datetime import datetime

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    HAS_DEPENDENCIES = False

# Constants
DIRS: List[Tuple[int, int, int]] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

# Image resolution constants for VMEvalKit
IMAGE_SIZE = (400, 400)  # VMEvalKit standard size
FIGURE_SIZE = (8, 8)     # Matplotlib figure size for rendering

Voxel = Tuple[int, int, int]


class RotationGenerator:
    """Self-contained 3D voxel mental rotation task generator."""
    
    def __init__(self):
        self.generated_positions = []
        
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate 3D mental rotation tasks (8-15 voxels, 30-90° or 90° rotations)."""
        print(f"🎯 Generating {num_tasks} 3D mental rotation tasks (8-15 voxels, limited angles)...")
        
        if not HAS_DEPENDENCIES:
            raise ImportError("NumPy, matplotlib, and PIL are required for rotation tasks")
        
        random.seed(42)
        np.random.seed(42)
        
        voxel_list = []
        max_attempts = num_tasks * 100
        attempts = 0
        
        while len(voxel_list) < num_tasks and attempts < max_attempts:
            attempts += 1
            try:
                # Use 8-15 voxels for challenging but viable shapes
                voxels = self._generate_snake(
                    N=np.random.randint(8, 16), 
                    Lmin=1, 
                    Lmax=3,
                    p_branch=0.2,  # Some branching for variety
                    max_deg=3, 
                    tries=1000
                )

                # Skip structures that are too simple
                # With more voxels, we want meaningful 3D structures
                if len(voxels) < 8:
                    continue

                # Mix of regular rotations and axis-aligned 90° rotations
                use_axis_aligned = random.choice([True, False])  # 50% chance of axis-aligned
                
                if use_axis_aligned:
                    # Use predefined axis-aligned rotations (90° increments)
                    (elev1, azim1), (elev2, azim2) = self._sample_axis_aligned_views()
                    angle_diff = 90.0  # Axis-aligned rotations are always 90°
                else:
                    # Generate random viewing angles with controlled separation
                    elev1, azim1 = self._sample_view()
                    elev2, azim2 = self._sample_view()
                    
                    # Ensure angle difference is between 30-90° for clear visualization
                    angle_diff = self._angle_between(elev1, azim1, elev2, azim2)
                    if angle_diff < 30 or angle_diff > 90:  # Accept 30-90 degree rotations
                        continue
                
                # Determine difficulty based on structure complexity and angle difference
                difficulty = self._assess_difficulty(voxels, angle_diff)
                
                task_data = {
                    "voxels": voxels,
                    "first_view": (elev1, azim1),
                    "final_view": (elev2, azim2),
                    "angle_difference": angle_diff,
                    "difficulty": difficulty,
                    "num_voxels": len(voxels)
                }
                
                voxel_list.append(task_data)
                print(f"✅ Generated task {len(voxel_list)}/{num_tasks}")
                
            except RuntimeError:
                continue  # Try again if generation failed
        
        if attempts >= max_attempts:
            print(f"Warning: Reached maximum attempts. Generated {len(voxel_list)} tasks.")
        
        self.generated_positions = voxel_list
        return voxel_list

    def _generate_snake(
        self,
        N: int = 16,
        Lmin: int = 2,
        Lmax: int = 5,
        p_branch: float = 0.35,
        max_deg: int = 3,
        tries: int = 500,
        rng: Optional[random.Random] = None,
    ) -> List[Voxel]:
        """Create a 3-D voxel snake."""
        if rng is None:
            rng = random.Random()

        for _ in range(tries):
            voxels: Set[Voxel] = {(0, 0, 0)}
            order: List[Voxel] = [(0, 0, 0)]
            axes_used: Set[str] = set()

            # Choose initial heading
            d = rng.choice(DIRS)
            axes_used.add(self._axis_of(d))

            while len(voxels) < N:
                # Grow main straight segment
                seg_len = min(rng.randint(Lmin, Lmax), N - len(voxels))
                x, y, z = order[-1]
                main_path: List[Voxel] = []

                for _ in range(seg_len):
                    x += d[0]; y += d[1]; z += d[2]
                    nxt = (x, y, z)

                    if nxt in voxels:
                        break
                    if self._neighbour_count(nxt, voxels) >= max_deg:
                        break
                    if any(
                        self._neighbour_count(nbr, voxels) + 1 > max_deg
                        for nbr in ((x + dx, y + dy, z + dz) for dx, dy, dz in DIRS)
                        if nbr in voxels
                    ):
                        break
                    main_path.append(nxt)
                else:
                    voxels.update(main_path)
                    order.extend(main_path)
                    axes_used.add(self._axis_of(d))

                if len(main_path) < seg_len:
                    break

                if len(voxels) >= N:
                    break

                # Optional branch from segment start
                if rng.random() < p_branch and len(voxels) < N and len(main_path) > 0:
                    seg_start_idx = len(order) - len(main_path) - 1
                    if seg_start_idx >= 0:
                        sx, sy, sz = order[seg_start_idx]
                        possible_branches = self._orthogonals(d)
                        if possible_branches:
                            branch_dir = rng.choice(possible_branches)
                            bx, by, bz = sx + branch_dir[0], sy + branch_dir[1], sz + branch_dir[2]
                            br_vox = (bx, by, bz)
                            if (
                                br_vox not in voxels
                                and self._neighbour_count(br_vox, voxels) < max_deg
                                and self._neighbour_count((sx, sy, sz), voxels) + 1 <= max_deg
                            ):
                                voxels.add(br_vox)
                                order.append(br_vox)
                                axes_used.add(self._axis_of(branch_dir))

                # Choose next heading
                orths = self._orthogonals(d)
                rng.shuffle(orths)

                unused_orths = [v for v in orths if self._axis_of(v) not in axes_used]
                for nd in unused_orths + orths:
                    tx, ty, tz = order[-1]
                    tx += nd[0]; ty += nd[1]; tz += nd[2]
                    if (tx, ty, tz) not in voxels:
                        d = nd
                        axes_used.add(self._axis_of(d))
                        break
                else:
                    break

            # Final acceptance test
            required_axes = {"x", "y", "z"}
            if len(voxels) == N and axes_used == required_axes:
                flipped = self._flip_voxels(order, axes=("x",))
                if not self._are_rotationally_equivalent(order, flipped):
                    return self._shift_to_origin(order)

        raise RuntimeError("Could not build a snake in the allotted attempts.")

    def _shift_to_origin(self, vox: List[Voxel]) -> List[Voxel]:
        """Shift voxel coordinates so the minimum is at origin."""
        if not vox:
            return []
        mins = [min(coord) for coord in zip(*vox)]
        dx, dy, dz = (-m for m in mins)
        return [(x + dx, y + dy, z + dz) for x, y, z in vox]

    def _orthogonals(self, d: Voxel) -> List[Voxel]:
        """Return the four directions that are perpendicular to `d`."""
        bad = {d, (-d[0], -d[1], -d[2])}
        return [v for v in DIRS if v not in bad]

    def _neighbour_count(self, v: Voxel, voxels: Set[Voxel]) -> int:
        """Number of occupied neighbours of voxel `v`."""
        x, y, z = v
        return sum((x + dx, y + dy, z + dz) in voxels for dx, dy, dz in DIRS)

    def _axis_of(self, d: Voxel) -> str:
        """Return 'x', 'y', or 'z' for a unit direction vector."""
        for idx, val in enumerate(d):
            if val != 0:
                return "xyz"[idx]
        raise ValueError("Invalid direction vector")

    def _flip_voxels(self, voxels: List[Voxel], axes: Tuple[str, ...] = ("x",)) -> List[Voxel]:
        """Flip voxels along specified axes."""
        if not voxels:
            return []

        xs, ys, zs = zip(*voxels)
        bounds = {
            "x": (min(xs), max(xs)),
            "y": (min(ys), max(ys)),
            "z": (min(zs), max(zs))
        }

        result = []
        for x, y, z in voxels:
            if "x" in axes:
                x = bounds["x"][1] - (x - bounds["x"][0])
            if "y" in axes:
                y = bounds["y"][1] - (y - bounds["y"][0])
            if "z" in axes:
                z = bounds["z"][1] - (z - bounds["z"][0])
            result.append((x, y, z))
        return result

    def _are_rotationally_equivalent(self, A: Iterable[Voxel], B: Iterable[Voxel]) -> bool:
        """Return True if set A can be rotated to match set B."""
        A, B = list(A), list(B)
        if len(A) != len(B):
            return False
            
        B_canon = self._canonicalize(B)
        rotations = self._rotation_matrices()
        
        for mat in rotations:
            A_rot = [self._apply_rotation(v, mat) for v in A]
            if self._canonicalize(A_rot) == B_canon:
                return True
        return False

    def _canonicalize(self, voxels: Iterable[Voxel]) -> Set[Voxel]:
        """Translate voxels so the minimal coordinate is at origin."""
        voxels = list(voxels)
        if not voxels:
            return set()
        anchor = min(voxels)
        ax, ay, az = anchor
        return {(x-ax, y-ay, z-az) for x, y, z in voxels}

    def _apply_rotation(self, v: Voxel, mat) -> Voxel:
        """Apply rotation matrix to a voxel."""
        x, y, z = v
        return (
            mat[0][0]*x + mat[0][1]*y + mat[0][2]*z,
            mat[1][0]*x + mat[1][1]*y + mat[1][2]*z,
            mat[2][0]*x + mat[2][1]*y + mat[2][2]*z,
        )

    def _rotation_matrices(self):
        """Generate the 24 orientation-preserving 3×3 rotation matrices."""
        from itertools import permutations, product
        
        mats = []
        for perm in permutations(range(3)):
            inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
            parity = inversions % 2

            for signs in product((1, -1), repeat=3):
                det = signs[0] * signs[1] * signs[2] * (-1)**parity
                if det == 1:
                    mat = [[0]*3 for _ in range(3)]
                    for row, (axis, s) in enumerate(zip(perm, signs)):
                        mat[row][axis] = s
                    mats.append(tuple(tuple(r) for r in mat))
        return mats

    def _sample_view(
        self,
        elev_range: Tuple[Tuple[int, int], ...] = ((15, 75), (105, 165), (195, 255), (285, 345)),
        azim_sectors: Tuple[Tuple[int, int], ...] = ((15, 75), (105, 165), (195, 255), (285, 345))
    ) -> Tuple[int, int]:
        """Return a random (elev, azim) viewing angle."""
        elev_sector = random.choice(elev_range)
        elev = random.uniform(*elev_sector)
        azim_sector = random.choice(azim_sectors)
        azim = random.uniform(*azim_sector)
        return int(elev), int(azim)
    
    def _sample_axis_aligned_views(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return two viewing angles that differ by exactly 90° around a clear axis."""
        # Predefined axis-aligned view pairs for easy visualization
        axis_aligned_pairs = [
            # 90° rotations around Z-axis (top view rotations)
            ((0, 0), (0, 90)),
            ((0, 45), (0, 135)),
            ((0, 90), (0, 180)),
            # 90° rotations around Y-axis (side view rotations)
            ((30, 0), (30, 90)),
            ((45, 0), (45, 90)),
            # Simple elevation changes with fixed azimuth
            ((30, 45), (60, 45)),
            ((45, 0), (75, 0)),
        ]
        return random.choice(axis_aligned_pairs)

    def _angle_between(self, elev1: float, azim1: float, elev2: float, azim2: float) -> float:
        """Calculate angle between two viewing directions in degrees."""
        e1, a1 = math.radians(elev1), math.radians(azim1)
        e2, a2 = math.radians(elev2), math.radians(azim2)

        v1 = (math.cos(e1) * math.cos(a1),
              math.cos(e1) * math.sin(a1),
              math.sin(e1))

        v2 = (math.cos(e2) * math.cos(a2),
              math.cos(e2) * math.sin(a2),
              math.sin(e2))

        dot = sum(p * q for p, q in zip(v1, v2))
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _assess_difficulty(self, voxels: List[Voxel], angle_diff: float) -> str:
        """Assess task difficulty based on structure complexity and rotation angle."""
        num_voxels = len(voxels)
        
        # Base complexity from number of voxels (8-15 range)
        # Scale: 8-9 voxels = 2 points, 10-12 = 4 points, 13-15 = 6 points
        if num_voxels <= 9:
            complexity_score = 2
        elif num_voxels <= 12:
            complexity_score = 4
        else:
            complexity_score = 6
        
        # Add complexity for structures spanning multiple axes
        axes_used = len(set(v[0] for v in voxels)) + len(set(v[1] for v in voxels)) + len(set(v[2] for v in voxels))
        if axes_used > 6:  # Structures spanning many coordinates
            complexity_score += 2
        elif axes_used > 4:
            complexity_score += 1
        
        # Factor in rotation angle
        if angle_diff == 90:  # Axis-aligned rotations are easier to visualize
            pass  # No complexity increase
        elif angle_diff > 60:
            complexity_score += 2
        elif angle_diff > 40:
            complexity_score += 1
        
        # Adjusted thresholds for 8-15 voxel range
        if complexity_score <= 4:
            return "easy"
        elif complexity_score <= 7:
            return "medium"  
        else:
            return "hard"


def generate_task_images(task_data: Dict[str, Any], task_id: str, base_dir: str) -> Tuple[str, str]:
    """
    Generate first and final frame images for a mental rotation task.
    
    Returns:
        (first_image_path, final_image_path)
    """
    if not HAS_DEPENDENCIES:
        raise ImportError("Required dependencies not available")
    
    voxels = task_data["voxels"]
    first_view = task_data["first_view"]
    final_view = task_data["final_view"]
    
    # Create temporary files that will be moved to per-question folders
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    first_temp_path = os.path.join(temp_dir, f"{task_id}_first.png")
    final_temp_path = os.path.join(temp_dir, f"{task_id}_final.png")
    
    # Generate first view image
    _render_voxel_image(voxels, first_view[0], first_view[1], first_temp_path)
    
    # Generate final view image  
    _render_voxel_image(voxels, final_view[0], final_view[1], final_temp_path)
    
    # Return temp paths that will be moved by create_dataset.py
    first_image_path = first_temp_path
    final_image_path = final_temp_path
    
    return first_image_path, final_image_path


def _render_voxel_image(voxels: List[Voxel], elev: float, azim: float, output_path: str):
    """Render voxel structure from specified viewing angle and save as image."""
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, subplot_kw={'projection': '3d'})
    
    # Plot cubes with proper lighting and styling
    _plot_cubes(voxels, ax, elev=elev, azim=azim)
    
    # Save as temporary high-res image
    temp_path = output_path.replace('.png', '_temp.png')
    fig.savefig(temp_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    
    # Process and resize to standard VMEvalKit format
    _process_and_save_image(temp_path, output_path, IMAGE_SIZE)


def _plot_cubes(
    positions: List[Voxel],
    ax,
    *,
    size: float = 1,
    elev: float = 25,
    azim: float = 35,
):
    """Draw cubes in 3D plot with proper styling for VMEvalKit."""
    # Generate cube faces with consistent coloring
    faces, colors = [], []
    for pos in positions:
        verts = _cube_vertices(pos, size)
        for face in _cube_faces(verts):
            faces.append(face)
            colors.append((0.7, 0.7, 0.9))  # Light blue color
    
    # Create 3D collection
    coll = Poly3DCollection(
        faces, 
        facecolors=colors, 
        linewidths=0.8, 
        edgecolors='black',
        alpha=0.8
    )
    ax.add_collection3d(coll)
    
    # Set up axes with proper scaling
    all_points = np.concatenate([_cube_vertices(p, size) for p in positions])
    _set_axes_equal(ax, all_points)
    
    # Configure view
    ax.set_proj_type('persp')
    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.view_init(elev=elev, azim=azim)


def _cube_vertices(origin: Voxel, size: float = 1) -> np.ndarray:
    """Generate vertices for a cube at given origin."""
    x, y, z = origin
    return np.array([
        [x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
        [x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size], [x, y + size, z + size],
    ])


def _cube_faces(verts: np.ndarray) -> List[List[np.ndarray]]:
    """Generate faces for a cube given its vertices."""
    return [
        [verts[j] for j in [0, 1, 2, 3]],  # bottom
        [verts[j] for j in [4, 5, 6, 7]],  # top
        [verts[j] for j in [0, 1, 5, 4]],  # front
        [verts[j] for j in [2, 3, 7, 6]],  # back
        [verts[j] for j in [1, 2, 6, 5]],  # right
        [verts[j] for j in [4, 7, 3, 0]],  # left
    ]


def _set_axes_equal(ax, pts: np.ndarray, padding: float = 0.2):
    """Force equal scaling on all axes with padding."""
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
    mid = pts.mean(axis=0)
    pad = max_range * padding
    
    for i, (set_lim, mid_val) in enumerate(zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid)):
        set_lim(mid_val - max_range - pad, mid_val + max_range + pad)


def _process_and_save_image(temp_path: str, final_path: str, image_size: Tuple[int, int]):
    """Process temporary image and save in VMEvalKit format."""
    image = Image.open(temp_path)
    
    # Crop to square if needed
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    
    # Resize to target size and convert to RGB
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    
    # Save final image
    image.save(final_path, "PNG")
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)


def generate_prompt(task_data: Dict[str, Any]) -> str:
    """Generate simplified text prompt for the mental rotation task."""
    num_voxels = task_data["num_voxels"]
    
    # Get the actual viewpoint positions
    elev1, azim1 = task_data["first_view"]
    elev2, azim2 = task_data["final_view"]
    
    # Specify exact camera/viewing angles
    prompt = (f"A {num_voxels}-block sculpture sits fixed on a table. "
              f"First frame: Your camera is at viewing angle (elevation: {elev1}°, azimuth: {azim1}°). "
              f"Final frame: Your camera is at viewing angle (elevation: {elev2}°, azimuth: {azim2}°). "
              f"Create a smooth video transition as the camera moves between these viewing angles, showing how the sculpture appears from different perspectives.")
    
    return prompt


def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a task pair in VMEvalKit format."""
    
    # Get base directory
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    
    # Generate images
    first_image_path, final_image_path = generate_task_images(task_data, task_id, base_dir)
    
    # Generate prompt
    prompt = generate_prompt(task_data)
    
    # Create task pair
    return {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": first_image_path,
        "final_image_path": final_image_path,
        "task_category": "3D Mental Rotation",
        "rotation_data": {
            "generation_method": "3D voxel snake with viewpoint rotation",
            "num_voxels": task_data["num_voxels"],
            "first_view_elev": task_data["first_view"][0],
            "first_view_azim": task_data["first_view"][1], 
            "final_view_elev": task_data["final_view"][0],
            "final_view_azim": task_data["final_view"][1],
            "angle_difference": task_data["angle_difference"],
            "structural_complexity": "snake_like_3d_voxels"
        },
        "difficulty": task_data["difficulty"],
        "created_at": datetime.now().isoformat()
    }


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create mental rotation dataset (8-15 voxels, 30-90° or 90° rotations)."""
    
    print(f"🎯 Creating 3D mental rotation dataset with {num_samples} samples...")
    
    # Generate tasks
    generator = RotationGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"rotation_{i:04d}"
        pair = create_task_pair(task_data, task_id)
        pairs.append(pair)
        print(f"✅ Created task {task_id}")
    
    # Create dataset
    dataset = {
        "name": "rotation_tasks",
        "description": f"3D mental rotation tasks (8-15 voxels, 30-90° or 90° rotations) for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Don't save to intermediate folder anymore - will be handled by create_dataset.py
    return dataset


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API
