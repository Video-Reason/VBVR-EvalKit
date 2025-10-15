# Constants for RAVEN task generation

IMAGE_SIZE = 160

# Drawing defaults
DEFAULT_LINE_WIDTH = 2

# Attribute value arrays
TYPE_VALUES = ["triangle", "square", "pentagon", "hexagon", "circle"]
SIZE_VALUES = [0.45, 0.55, 0.65, 0.75, 0.85]
COLOR_VALUES = [0, 28, 56, 84, 112, 140, 168, 196, 224, 255]
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135]  # Rotation angles in degrees
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Number of entities
UNI_VALUES = [False, True]  # Uniformity (all entities same or different)

# Min/max level indices for attributes
TYPE_MIN = 0
TYPE_MAX = len(TYPE_VALUES) - 1

SIZE_MIN = 0  
SIZE_MAX = len(SIZE_VALUES) - 1

COLOR_MIN = 0
COLOR_MAX = len(COLOR_VALUES) - 1

ANGLE_MIN = 0
ANGLE_MAX = len(ANGLE_VALUES) - 1

NUM_MIN = 0
NUM_MAX = len(NUM_VALUES) - 1

UNI_MIN = 0
UNI_MAX = len(UNI_VALUES) - 1

# Convenience clamp function
def clamp(value, lo, hi):
    return max(lo, min(hi, value))