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
    'mirror_clock': {
        'name': 'Mirror Clock',
        'description': 'Spatial reasoning and mirror transformation using analog clock reflections',
        'module': 'vmevalkit.tasks.mirror_clock_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'light_sequence': {
        'name': 'Light Sequence',
        'description': 'Spatial reasoning and mathematical pattern recognition with light sequences',
        'module': 'vmevalkit.tasks.light_sequence_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}

