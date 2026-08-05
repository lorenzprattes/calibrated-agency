"""Optional model-serving backends.

Import concrete backends from their modules so importing the core never requires vLLM,
Transformers, or PyTorch.
"""

from .base import ActionScorer, ContinuationResult

__all__ = ["ActionScorer", "ContinuationResult"]
