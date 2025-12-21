"""
Task Catalog for VMEvalKit - Registry of all available reasoning tasks.
"""

TASK_REGISTRY = {
    'videothinkbench': {
        'name': 'VideoThinkBench',
        'description': 'Complete VideoThinkBench dataset with all reasoning tasks (~4.1k tasks)',
        'module': 'vmevalkit.tasks.external.videothinkbench_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'chess': {
        'name': 'Chess',
        'description': 'Strategic thinking and tactical pattern recognition',
        'module': 'vmevalkit.tasks.chess_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'maze': {
        'name': 'Maze',
        'description': 'Spatial reasoning and navigation planning',
        'module': 'vmevalkit.tasks.maze_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'raven': {
        'name': 'RAVEN',
        'description': 'Abstract reasoning and pattern completion',
        'module': 'vmevalkit.tasks.raven_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'rotation': {
        'name': 'Rotation',
        'description': '3D mental rotation and spatial visualization',
        'module': 'vmevalkit.tasks.rotation_task.rotation_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'sudoku': {
        'name': 'Sudoku',
        'description': 'Logical reasoning and constraint satisfaction',
        'module': 'vmevalkit.tasks.sudoku_task.sudoku_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'clock': {
        'name': 'Clock',
        'description': 'Temporal reasoning and time calculation',
        'module': 'vmevalkit.tasks.clock_task.clock_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'arc_agi_2': {
        'name': 'ARC AGI 2',
        'description': 'ARC AGI reasoning and problem solving',
        'module': 'vmevalkit.tasks.external.videothinkbench_arc_agi_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'eyeballing_puzzles': {
        'name': 'Eyeballing Puzzles',
        'description': 'Eyeballing puzzles reasoning and problem solving',
        'module': 'vmevalkit.tasks.external.videothinkbench_eyeballing_puzzles_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'visual_puzzles': {
        'name': 'Visual Puzzles',
        'description': 'Visual puzzles reasoning and problem solving',
        'module': 'vmevalkit.tasks.external.videothinkbench_visual_puzzles_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'text_centric_tasks': {
        'name': 'Text Centric Tasks',
        'description': 'Mathematical reasoning and multimodal understanding',
        'module': 'vmevalkit.tasks.external.videothinkbench_text_centric_tasks_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'object_subtraction': {
        'name': 'Object Subtraction',
        'description': 'Selective object removal with multi-level cognitive reasoning',
        'module': 'vmevalkit.tasks.object_subtraction_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'majority_color': {
        'name': 'Majority Color',
        'description': 'Recolor all objects to match the most common color in the scene',
        'module': 'vmevalkit.tasks.majority_color_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'grid_shift': {
        'name': 'Grid Shift',
        'description': 'Shift all blocks 1-2 steps in a random direction within a 6x6 grid',
        'module': 'vmevalkit.tasks.grid_shift_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'shape_sorter': {
        'name': 'Shape Sorter',
        'description': '2D shape matching under a fixed top-down camera',
        'module': 'vmevalkit.tasks.shape_sorter_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'object_rearr': {
        'name': 'Object Rearrangement',
        'description': 'Spatial reasoning and object manipulation with spatial relations',
        'module': 'vmevalkit.tasks.object_rearr_task.object_rearr',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'sliding_puzzle': {
        'name': 'Sliding Puzzle',
        'description': 'Spatial reasoning and simple planning through near-complete sliding puzzles',
        'module': 'vmevalkit.tasks.sliding_puzzle_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'mme_cof': {
        'name': 'MME-CoF',
        'description': 'Video Chain-of-Frame reasoning evaluation across 16 cognitive domains (59 tasks)',
        'module': 'vmevalkit.tasks.external.mme_cof_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'perspective_taking': {
        'name': 'Perspective Taking',
        'description': 'Spatial reasoning and viewpoint transformation from agent perspective',
        'module': 'vmevalkit.tasks.perspective_taking_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'edit_distance': {
        'name': 'Edit Distance',
        'description': 'String edit distance calculation and numerical reasoning',
        'module': 'vmevalkit.tasks.edit_distance_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'object_permanence': {
        'name': 'Object Permanence',
        'description': 'Object permanence reasoning - objects remain unchanged when occluder moves',
        'module': 'vmevalkit.tasks.object_permanence_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'control_panel': {
        'name': 'Control Panel',
        'description': 'Control panel animation - lever position determines indicator light color',
        'module': 'vmevalkit.tasks.control_panel_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'vpct': {
        'name': 'VPCT',
        'description': 'Visual Physics Comprehension Test - predict which bucket the ball will fall into',
        'module': 'vmevalkit.tasks.external.vpct_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'camelCase12/vpct-1',
        'hf_special_format': True  # Indicates file-based format, not standard dataset format
    },
    '2d_dice_reasoning': {
        'name': '2D Dice Reasoning',
        'description': '2D dice opposite face spatial reasoning and logical deduction (opposite faces sum to 7)',
        'module': 'vmevalkit.tasks.dice_2d_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'tetris': {
        'name': 'Tetris',
        'description': 'Tetris line-clearing reasoning tasks with animation instructions',
        'module': 'vmevalkit.tasks.tetris_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'mirror_clock': {
        'name': 'Mirror Clock',
        'description': 'Spatial reasoning and mirror transformation using analog clock reflections',
        'module': 'vmevalkit.tasks.mirror_clock_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'counting_objects': {
        'name': 'Counting Circles',
        'description': 'Visual counting of Objects',
        'module': 'vmevalkit.tasks.counting_objects_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'letter_counting': {
        'name': 'Letter Counting',
        'description': 'Count letter occurrences in words',
        'module': 'vmevalkit.tasks.letter_counting_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'subway_pathfinding': {
        'name': 'Subway Pathfinding',
        'description': 'Navigate through subway networks from source to destination',
        'module': 'vmevalkit.tasks.subway_pathfinding_task',
    },
    'light_sequence': {
        'name': 'Light Sequence',
        'description': 'Spatial reasoning and mathematical pattern recognition with light sequences',
        'module': 'vmevalkit.tasks.light_sequence_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'dot_to_dot': {
        'name': 'Dot to Dot',
        'description': 'Sequential dot connection to reveal patterns',
        'module': 'vmevalkit.tasks.dot_to_dot_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'symmetry_completion': {
        'name': 'Symmetry Completion',
        'description': 'Left-right symmetry reasoning - complete right half by mirroring left half',
        'module': 'vmevalkit.tasks.symmetry_completion_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'tower_of_hanoi': {
        'name': 'Tower of Hanoi',
        'description': 'Single-move planning and constraint satisfaction',
        'module': 'vmevalkit.tasks.tower_of_hanoi_task.tower_of_hanoi_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'rotation_puzzle': {
        'name': 'Rotation Puzzle',
        'description': 'Pipe puzzle with rotatable squares to connect paths',
        'module': 'vmevalkit.tasks.rotation_puzzle_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}
