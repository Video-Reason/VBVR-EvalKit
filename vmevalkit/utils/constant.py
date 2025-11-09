DOMAIN_REGISTRY = {
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
        'create_function': 'create_dataset',  # Standard function like other domains
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
        'name': 'Arc AGI',
        'description': 'Arc AGI reasoning and problem solving',
        'module': 'vmevalkit.tasks.arc_agi_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'ARC_AGI_2',
        'hf_split': 'test',
        'hf_prompt_column': 'prompt',
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',
    },
    'eyeballing_puzzles': {
        'name': 'Eyeballing Puzzles',
        'description': 'Eyeballing puzzles reasoning and problem solving',
        'module': 'vmevalkit.tasks.eyeballing_puzzles_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Eyeballing_Puzzles',
        'hf_split': 'test',
        'hf_prompt_column': 'prompt',
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',
    },
    'visual_puzzles': {
        'name': 'Visual Puzzles',
        'description': 'Visual puzzles reasoning and problem solving',
        'module': 'vmevalkit.tasks.visual_puzzles_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Visual_Puzzles',
        'hf_split': 'test',
        'hf_prompt_column': 'prompt',
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',
    }
}