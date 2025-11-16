"""
MME-CoF Chain-of-Frame Reasoning Prompts

Based on the MME-CoF benchmark for evaluating video models as zero-shot reasoners.
These prompts encourage frame-by-frame reasoning evolution in generated videos.
"""

# Category-specific prompts that guide chain-of-frame reasoning
CATEGORY_PROMPTS = {
    "2D_geometry_reasoning": {
        "prefix": "Animate the geometric transformation step-by-step:",
        "instruction": "Show the progressive geometric evolution with clear intermediate steps visible in each frame."
    },
    "3D_geometry_reasoning": {
        "prefix": "Demonstrate the 3D spatial reasoning process:",
        "instruction": "Reveal the three-dimensional transformation gradually, showing depth and perspective changes frame by frame."
    },
    "abstract_reasoning": {
        "prefix": "Visualize the abstract pattern evolution:",
        "instruction": "Display the logical progression of abstract concepts with clear reasoning steps in the animation."
    },
    "chess": {
        "prefix": "Animate the chess move sequence:",
        "instruction": "Show each chess move step-by-step with pieces moving sequentially to demonstrate the tactical progression."
    },
    "common_sense_reasoning": {
        "prefix": "Demonstrate the common sense scenario:",
        "instruction": "Animate the logical sequence of events that follows natural cause-and-effect relationships."
    },
    "counting_reasoning": {
        "prefix": "Visualize the counting process:",
        "instruction": "Show the counting or quantity change step-by-step with clear visual indicators in each frame."
    },
    "logical_reasoning": {
        "prefix": "Animate the logical deduction sequence:",
        "instruction": "Display each step of the logical reasoning chain with clear cause-and-effect relationships."
    },
    "physics_reasoning": {
        "prefix": "Simulate the physical process:",
        "instruction": "Show the physics-based transformation with realistic motion and force interactions visible frame by frame."
    },
    "physics_based_reasoning": {
        "prefix": "Simulate the physical process:",
        "instruction": "Show the physics-based transformation with realistic motion and force interactions visible frame by frame."
    },
    "practical_reasoning": {
        "prefix": "Demonstrate the practical solution:",
        "instruction": "Animate the step-by-step practical approach to solving the problem."
    },
    "visual_analogy_reasoning": {
        "prefix": "Show the visual analogy transformation:",
        "instruction": "Animate how the pattern transforms analogously with clear correspondence between elements."
    },
    "visual_arithmetic_reasoning": {
        "prefix": "Visualize the arithmetic operation:",
        "instruction": "Show the mathematical operation step-by-step with visual representations of the calculation process."
    },
    "visual_trace_reasoning": {
        "prefix": "Animate the path or trace:",
        "instruction": "Show the sequential tracing or path-following process with clear progression through each step."
    },
    "rotation_reasoning": {
        "prefix": "Animate the rotation transformation:",
        "instruction": "Show the object rotating step-by-step with clear intermediate orientations visible in each frame."
    },
    "real_world_spatial_reasoning": {
        "prefix": "Demonstrate the real-world spatial scenario:",
        "instruction": "Show the spatial relationships and transformations as they would occur in the real world."
    },
    "table_and_chart_reasoning": {
        "prefix": "Visualize the data transformation:",
        "instruction": "Show how the data in tables or charts changes step-by-step to reach the solution."
    },
    "visual_detail_reasoning": {
        "prefix": "Highlight the visual details:",
        "instruction": "Show the important visual details and how they lead to the solution step-by-step."
    }
}

def get_prompt_for_category(category: str) -> str:
    """
    Generate a chain-of-frame reasoning prompt for a given category.
    
    Args:
        category: The reasoning category (e.g., "2D_geometry_reasoning")
    
    Returns:
        A complete prompt string that encourages frame-by-frame reasoning
    """
    if category not in CATEGORY_PROMPTS:
        # Dynamically generate prompt for unknown categories
        category_readable = category.replace('_', ' ').replace('  ', ' ').title()
        return (
            f"Animate this {category_readable} task step-by-step: "
            f"Show the {category_readable.lower()} process with clear intermediate steps "
            "and logical progression in each frame of the video."
        )
    
    prompt_config = CATEGORY_PROMPTS[category]
    return f"{prompt_config['prefix']} {prompt_config['instruction']}"


def get_category_description(category: str) -> str:
    """Get a human-readable description of the reasoning category."""
    descriptions = {
        "2D_geometry_reasoning": "2D geometric transformation and spatial reasoning",
        "3D_geometry_reasoning": "3D spatial reasoning and perspective understanding",
        "abstract_reasoning": "Abstract pattern recognition and logical thinking",
        "chess": "Strategic chess reasoning and tactical planning",
        "common_sense_reasoning": "Real-world common sense understanding",
        "counting_reasoning": "Counting and quantity-based reasoning",
        "logical_reasoning": "Formal logical deduction and inference",
        "physics_reasoning": "Physical simulation and causality understanding",
        "physics_based_reasoning": "Physical simulation and causality understanding",
        "practical_reasoning": "Practical problem-solving and real-world application",
        "visual_analogy_reasoning": "Visual pattern analogies and relationships",
        "visual_arithmetic_reasoning": "Visual mathematical operations and calculations",
        "visual_trace_reasoning": "Path tracing and sequential navigation",
        "rotation_reasoning": "Mental rotation and orientation transformation",
        "real_world_spatial_reasoning": "Real-world spatial relationships and navigation",
        "table_and_chart_reasoning": "Data interpretation from tables and charts",
        "visual_detail_reasoning": "Fine-grained visual detail observation and analysis"
    }
    return descriptions.get(category, category.replace('_', ' ').title())

