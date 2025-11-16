DOMAIN_REGISTRY = {
    'videothinkbench': {
        'name': 'VideoThinkBench',
        'description': 'Complete VideoThinkBench dataset with all reasoning tasks (~4.1k tasks)',
        'module': 'vmevalkit.tasks.videothinkbench_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_meta': True,  # Special flag for meta-task that downloads all subsets
        'hf_subsets': ['arc_agi_2', 'eyeballing_puzzles', 'visual_puzzles', 'mazes', 'text_centric_tasks']
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
    },
    'mazes': {
        'name': 'Mazes',
        'description': 'Path-finding and navigation challenges',
        'module': 'vmevalkit.tasks.mazes_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Mazes',
        'hf_split': 'test',
        'hf_prompt_column': 'prompt',
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',
    },
    'text_centric_tasks': {
        'name': 'Text Centric Tasks',
        'description': 'Mathematical reasoning and multimodal understanding',
        'module': 'vmevalkit.tasks.text_centric_tasks_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'hf': True,
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Text_Centric_Tasks',
        'hf_split': 'test',
        'hf_prompt_column': 'prompt',
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',
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
        'description': 'Video Chain-of-Frame reasoning evaluation across 12 cognitive domains (59 tasks)',
        'module': 'vmevalkit.tasks.mme_cof_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs'],
        'emoji': '🎬',
        'hf': True,
        'hf_dataset': 'YourOrg/MME-CoF-VMEval',  # TODO: Update with your HuggingFace org/dataset name
        'hf_subset': None,
        'hf_split': 'train',
        'hf_domain': 'mme_cof',
        'hf_task_id_prefix': 'mme_cof',
        'hf_prompt_column': 'prompt',  # Generated prompts
        'hf_image_column': 'image',
        'hf_solution_image_column': 'solution_image',  # LLM-generated solutions
        'hf_category_column': 'category',  # Reasoning category metadata
    }
}