# Rule implementations for RAVEN task generation

import copy
from typing import Optional


class Rule:
    """Base class for RAVEN rules."""
    def __init__(self, name: str, attr: str, value: int, component_idx: int):
        self.name = name  # Rule type name
        self.attr = attr  # Attribute to apply rule to
        self.value = value  # Rule parameter (e.g., step size)
        self.component_idx = component_idx

    def apply_rule(self, aot, in_aot=None):
        """Apply rule to transform AoT."""
        raise NotImplementedError


class Constant(Rule):
    """Constant rule - no change between panels."""
    def __init__(self, attr: str, component_idx: int):
        super().__init__("Constant", attr, 0, component_idx)
    
    def apply_rule(self, aot, in_aot=None):
        """Return a copy without changes."""
        return copy.deepcopy(in_aot or aot)


class Progression(Rule):
    """Progression rule - incremental change across panels."""
    def __init__(self, attr: str, value: int, component_idx: int):
        super().__init__("Progression", attr, value, component_idx)
    
    def apply_rule(self, aot, in_aot=None):
        """Apply progression to the specified attribute."""
        base = aot
        out = copy.deepcopy(in_aot or aot)
        
        # Navigate to the correct component following RAVEN structure
        # Root -> Structure -> Component -> Layout -> Entity
        component = out.children[0].children[self.component_idx]
        layout = component.children[0]
        
        if self.attr == "Number":
            # Change number of entities
            current_num = layout.number.get_value_level()
            new_level = current_num + self.value
            new_level = max(layout.number.min_level, min(new_level, layout.number.max_level))
            layout.number.set_value_level(new_level)
            # Resample entities with new number
            layout._resample(True)
            
        elif self.attr == "Position":
            # Rotate positions
            if layout.position.value_idx is not None:
                new_idx = []
                for idx in layout.position.value_idx:
                    new_idx.append((idx + self.value) % len(layout.position.values))
                layout.position.set_value_idx(new_idx)
                # Update entity positions
                pos = layout.position.get_value()
                for i, entity in enumerate(layout.children):
                    if i < len(pos):
                        entity.bbox = pos[i]
                        
        elif self.attr == "Type":
            # Progress entity types
            for entity in layout.children:
                current = entity.type.get_value_level()
                new_level = current + self.value
                new_level = max(entity.type.min_level, min(new_level, entity.type.max_level))
                entity.type.set_value_level(new_level)
                
        elif self.attr == "Size":
            # Progress entity sizes
            for entity in layout.children:
                current = entity.size.get_value_level()
                new_level = current + self.value
                new_level = max(entity.size.min_level, min(new_level, entity.size.max_level))
                entity.size.set_value_level(new_level)
                
        elif self.attr == "Color":
            # Progress entity colors
            for entity in layout.children:
                current = entity.color.get_value_level()
                new_level = current + self.value
                new_level = max(entity.color.min_level, min(new_level, entity.color.max_level))
                entity.color.set_value_level(new_level)
        
        return out


class Arithmetic(Rule):
    """Arithmetic rule - additive/subtractive operations."""
    def __init__(self, attr: str, value: int, component_idx: int):
        super().__init__("Arithmetic", attr, value, component_idx)
    
    def apply_rule(self, aot, in_aot=None):
        """Apply arithmetic operation."""
        # For now, treat as progression for simplicity
        # Full arithmetic implementation would combine attributes from multiple panels
        return Progression(self.attr, self.value, self.component_idx).apply_rule(aot, in_aot)


class Distribute_Three(Rule):
    """Distribute Three rule - three different values across panels."""
    def __init__(self, attr: str, component_idx: int):
        super().__init__("Distribute_Three", attr, 0, component_idx)
    
    def apply_rule(self, aot, in_aot=None):
        """Apply distribute three rule."""
        # For now, treat as constant for simplicity
        # Full implementation would ensure three distinct values
        return copy.deepcopy(in_aot or aot)


def Rule_Wrapper(name: str, attr: str, param, component_idx: int):
    """Factory function to create rule instances."""
    if name == "Constant":
        return Constant(attr, component_idx)
    elif name == "Progression":
        value = param if isinstance(param, int) else 1
        return Progression(attr, value, component_idx)
    elif name == "Arithmetic":
        value = param if isinstance(param, int) else 1
        return Arithmetic(attr, value, component_idx)
    elif name == "Distribute_Three":
        return Distribute_Three(attr, component_idx)
    else:
        # Fallback to Constant for unknown rules
        return Constant(attr, component_idx)