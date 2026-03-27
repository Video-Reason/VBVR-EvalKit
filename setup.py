#!/usr/bin/env python3
"""
VBVR-Bench
Setup script for package installation.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "VBVR-Bench"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'Pillow>=8.0.0',
        'tqdm>=4.60.0',
    ]

setup(
    name='vbvr-bench',
    version='0.1.0',
    author='VBVR-Bench Team',
    description='A rule-based evaluation toolkit for assessing video generation models on 100 visual reasoning tasks. Each task is evaluated by a dedicated rule-based evaluator.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Video-Reason/VBVR-Bench',
    packages=find_packages(),
    py_modules=['evaluate', 'run_evaluation'],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
        'gpu': [
            'torch>=1.9.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'vbvr-evaluate=evaluate:main',
            'vbvr-run-evaluation=run_evaluation:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords='video-generation evaluation benchmark visual-reasoning',
)
