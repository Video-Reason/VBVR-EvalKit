"""Evaluation module for VMEvalKit.

This module contains various evaluation methods for assessing video generation models'
reasoning capabilities.

Core evaluators (lazy-loaded due to heavy dependencies):
- HumanEvaluator: Gradio-based human evaluation interface
- GPT4OEvaluator: Single-frame evaluation using GPT-4O vision
- InternVLEvaluator: Automated evaluation using local VLM
- MultiFrameEvaluator: Generic multi-frame evaluator wrapper (works with any base evaluator)

Multi-frame evaluation components (always available):
- FrameSampler: Hybrid frame sampling from video
- FrameConsistencyAnalyzer: Frame-to-frame consistency analysis
- VotingAggregator: Weighted voting for score aggregation
- MultiFrameEvaluationPipeline: Complete multi-frame evaluation workflow
"""

import importlib
from typing import TYPE_CHECKING

# Multi-frame evaluation components (lightweight, always available)
from .frame_sampler import FrameSampler, SampledFrame
from .consistency import (
    FrameConsistencyAnalyzer,
    ConsistencyResult,
    SimilarityMetric,
    compute_frame_weights,
    compute_stability_score
)
from .voting import (
    VotingAggregator,
    VotingResult,
    VotingMethod,
    FrameScore,
    MultiFrameEvaluationPipeline,
    aggregate_scores
)

# Type hints for lazy-loaded classes
if TYPE_CHECKING:
    from .human_eval import HumanEvaluator
    from .gpt4o_eval import GPT4OEvaluator
    from .internvl import InternVLEvaluator
    from .multiframe_eval import MultiFrameEvaluator


# Lazy loading for evaluators with heavy dependencies
_LAZY_MODULES = {
    'HumanEvaluator': '.human_eval',
    'GPT4OEvaluator': '.gpt4o_eval',
    'InternVLEvaluator': '.internvl',
    'MultiFrameEvaluator': '.multiframe_eval',
}


def __getattr__(name: str):
    """Lazy load evaluators with heavy dependencies."""
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core evaluators (lazy-loaded)
    'HumanEvaluator',
    'GPT4OEvaluator',
    'InternVLEvaluator',
    'MultiFrameEvaluator',
    # Frame sampling
    'FrameSampler',
    'SampledFrame',
    # Consistency analysis
    'FrameConsistencyAnalyzer',
    'ConsistencyResult',
    'SimilarityMetric',
    'compute_frame_weights',
    'compute_stability_score',
    # Voting aggregation
    'VotingAggregator',
    'VotingResult',
    'VotingMethod',
    'FrameScore',
    'MultiFrameEvaluationPipeline',
    'aggregate_scores',
]
