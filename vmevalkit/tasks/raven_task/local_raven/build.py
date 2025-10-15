# Build AoT trees for different RAVEN configurations

from typing import List, Tuple

from .aot import Root, Structure, Component, Layout
from .constraints import gen_layout_constraint, gen_entity_constraint


def _grid_positions(grid_rows: int, grid_cols: int) -> List[Tuple[float, float, float, float]]:
    """Generate grid positions for distribute configurations."""
    positions = []
    cell_h = 1.0 / grid_rows
    cell_w = 1.0 / grid_cols
    for r in range(grid_rows):
        for c in range(grid_cols):
            y = (r + 0.5) * cell_h
            x = (c + 0.5) * cell_w
            # Use smaller size for entities in grid
            positions.append((y, x, min(cell_h, cell_w) * 0.4, min(cell_h, cell_w) * 0.4))
    return positions


def build_center_single() -> Root:
    """Build AoT for center_single configuration."""
    # Single centered position
    positions = [(0.5, 0.5, 0.5, 0.5)]
    
    # Create constraints
    layout_constraint = gen_layout_constraint("planar", positions, 
                                               num_min=0, num_max=0,  # Single entity
                                               uni_min=0, uni_max=1)
    entity_constraint = gen_entity_constraint()
    
    # Build tree
    layout = Layout("center", layout_constraint, entity_constraint)
    comp = Component("singleton")
    comp.insert(layout)
    struct = Structure("Singleton")
    struct.insert(comp)
    root = Root("center_single")
    root.insert(struct)
    
    return root


def build_distribute_four() -> Root:
    """Build AoT for distribute_four configuration (2x2 grid)."""
    positions = _grid_positions(2, 2)
    
    # Create constraints
    layout_constraint = gen_layout_constraint("planar", positions,
                                               num_min=0, num_max=len(positions)-1,  # 1-4 entities
                                               uni_min=0, uni_max=1)
    entity_constraint = gen_entity_constraint()
    
    # Build tree
    layout = Layout("2x2", layout_constraint, entity_constraint)
    comp = Component("grid")
    comp.insert(layout)
    struct = Structure("Grid_2x2")
    struct.insert(comp)
    root = Root("distribute_four")
    root.insert(struct)
    
    return root


def build_distribute_nine() -> Root:
    """Build AoT for distribute_nine configuration (3x3 grid)."""
    positions = _grid_positions(3, 3)
    
    # Create constraints
    layout_constraint = gen_layout_constraint("planar", positions,
                                               num_min=0, num_max=len(positions)-1,  # 1-9 entities
                                               uni_min=0, uni_max=1)
    entity_constraint = gen_entity_constraint()
    
    # Build tree
    layout = Layout("3x3", layout_constraint, entity_constraint)
    comp = Component("grid")
    comp.insert(layout)
    struct = Structure("Grid_3x3")
    struct.insert(comp)
    root = Root("distribute_nine")
    root.insert(struct)
    
    return root