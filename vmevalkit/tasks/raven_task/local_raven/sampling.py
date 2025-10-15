import random
from typing import List

from .rules import Rule_Wrapper


def sample_rules(max_components: int = 1, configuration: str = None) -> List[list]:
    """Simple rule sampler focused on visual attribute progressions.
    Returns list[component][rules]. Prioritizes Type and Size changes for clear patterns.
    EXCLUDES Color progressions to maintain visual clarity.
    """
    component_idx = 0
    
    # Focus on visual attributes that create clear progressions for all configurations
    # Exclude Color to maintain visual clarity - shapes should remain easily visible
    # Avoid Number/Position rules that can cause empty panels with single entities
    main_attr = random.choice(["Type", "Size"])
    rules = [Rule_Wrapper("Progression", main_attr, 1, component_idx)]
    
    # Add a second attribute for variety (but never Color)
    remaining_attrs = [attr for attr in ["Type", "Size"] if attr != main_attr]
    if remaining_attrs:
        extra_attr = random.choice(remaining_attrs)
        rules.append(Rule_Wrapper("Progression", extra_attr, 1, component_idx))
    
    # Could optionally add a Constant rule for Color to ensure it stays the same
    # rules.append(Rule_Wrapper("Constant", "Color", 0, component_idx))
    
    return [rules]


