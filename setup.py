"""
Setup script for VMEvalKit.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith("#")]

setup(
    name="vmevalkit",
    version="0.1.0",
    author="VMEvalKit Team",
    author_email="hokinxqdeng@gmail.com",
    description="A unified inference and evaluation framework for video generation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hokindeng/VMEvalKit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.1.0",
            "mypy>=1.7.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
        # docs extras removed - no documentation build needed
    },
    # CLI entry point removed - use examples/ scripts instead
    include_package_data=True,
    # No package data needed - using external data sources only
    project_urls={
        "Bug Reports": "https://github.com/hokindeng/VMEvalKit/issues",
        "Documentation": "https://vmevalkit.readthedocs.io",
        "Source": "https://github.com/hokindeng/VMEvalKit",
    },
)
