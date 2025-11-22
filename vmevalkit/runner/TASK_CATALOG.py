"""
Task Catalog for VMEvalKit - Registry of all available reasoning tasks.

Pure registry with no imports or logic - just task definitions organized by type.
Uses string module paths for flexible dynamic loading.
"""

# ========================================
# TASK REGISTRY
# ========================================

DOMAIN_REGISTRY = {
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
    }
}

