import numpy as np
import cv2
from typing import List

from .aot import Root
from .const import IMAGE_SIZE, DEFAULT_LINE_WIDTH


def _draw_triangle(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = np.array([
        [cy - r, cx],
        [cy + r, cx - int(0.866 * r)],
        [cy + r, cx + int(0.866 * r)],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_square(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pt1 = (cx - r, cy - r)
    pt2 = (cx + r, cy + r)
    cv2.rectangle(img, pt1, pt2, color, thickness)


def _draw_pentagon(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = []
    for k in range(5):
        theta = -np.pi / 2 + 2 * np.pi * k / 5
        pts.append([cy + int(r * np.sin(theta)), cx + int(r * np.cos(theta))])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_hexagon(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    pts = []
    for k in range(6):
        theta = -np.pi / 2 + 2 * np.pi * k / 6
        pts.append([cy + int(r * np.sin(theta)), cx + int(r * np.cos(theta))])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_circle(img, center, size, color, thickness):
    cy, cx = center
    r = int(size * min(img.shape) / 2)
    cv2.circle(img, (cx, cy), r, color, thickness)


def render_panel(root: Root) -> np.ndarray:
    """Render a panel following original RAVEN approach."""
    canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
    structure_name, entities = root.prepare()
    structure_img = render_structure(structure_name)
    background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    # Render each entity
    for entity in entities:
        entity_img = render_entity(entity)
        background = layer_add(background, entity_img)
    
    # Add structure rendering
    background = layer_add(background, structure_img)
    
    # Return inverted (original RAVEN uses canvas - background)
    return canvas - background


def render_structure(structure_name: str) -> np.ndarray:
    """Render structure lines (for Left_Right, Up_Down, etc)."""
    ret = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if structure_name == "Left_Right":
        ret[:, int(0.5 * IMAGE_SIZE)] = 255
    elif structure_name == "Up_Down":
        ret[int(0.5 * IMAGE_SIZE), :] = 255
    # Singleton and others have no structure lines
    return ret


def render_entity(entity) -> np.ndarray:
    """Render individual entity following original RAVEN approach."""
    y, x, h, w = entity.bbox
    entity_type = entity.type.get_value()
    entity_size = entity.size.get_value()
    entity_color = entity.color.get_value()
    
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    # Convert position to pixel coordinates (center)
    center = (int(x * IMAGE_SIZE), int(y * IMAGE_SIZE))
    
    # Calculate size
    unit = min(h, w) * IMAGE_SIZE / 2
    color = 255 - entity_color  # Invert color
    
    if entity_type == "triangle":
        size = min(h, w)
        _draw_triangle(img, center, size, color, DEFAULT_LINE_WIDTH)
    elif entity_type == "square":
        size = min(h, w) 
        _draw_square(img, center, size, color, DEFAULT_LINE_WIDTH)
    elif entity_type == "pentagon":
        size = min(h, w)
        _draw_pentagon(img, center, size, color, DEFAULT_LINE_WIDTH)
    elif entity_type == "hexagon":
        size = min(h, w)
        _draw_hexagon(img, center, size, color, DEFAULT_LINE_WIDTH)
    elif entity_type == "circle":
        size = min(h, w)
        _draw_circle(img, center, size, color, DEFAULT_LINE_WIDTH)
    
    return img


def layer_add(lower_layer: np.ndarray, higher_layer: np.ndarray) -> np.ndarray:
    """Add layers following original RAVEN approach."""
    lower_layer[higher_layer > 0] = 0
    return lower_layer + higher_layer


def generate_matrix(array_list: List[np.ndarray]) -> np.ndarray:
    """Generate 3x3 matrix from panel images, following original RAVEN implementation."""
    assert len(array_list) <= 9
    img_grid = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3), np.uint8)
    for idx in range(len(array_list)):
        i, j = divmod(idx, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = array_list[idx]
    
    # Draw grid lines exactly like original RAVEN
    for x in [0.33, 0.67]:
        img_grid[int(x * IMAGE_SIZE * 3) - 1:int(x * IMAGE_SIZE * 3) + 1, :] = 0
    for y in [0.33, 0.67]:
        img_grid[:, int(y * IMAGE_SIZE * 3) - 1:int(y * IMAGE_SIZE * 3) + 1] = 0
    
    return img_grid


