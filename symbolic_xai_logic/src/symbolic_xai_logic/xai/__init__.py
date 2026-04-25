from __future__ import annotations
from .base import Explainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .lrp import LRPExplainer
from .rule_extraction import RuleExtractor
from .concept_probe import ConceptProbe
from .symbolic_regression import SymbolicRegressionExplainer

XAI_REGISTRY: dict[str, type] = {
    "lime": LIMEExplainer,
    "shap": SHAPExplainer,
    "lrp": LRPExplainer,
    "rule_extraction": RuleExtractor,
    "concept_probe": ConceptProbe,
    "symbolic_regression": SymbolicRegressionExplainer,
}


def get_explainer(name: str, model, game, **kwargs) -> "Explainer":
    if name not in XAI_REGISTRY:
        raise ValueError(f"Unknown explainer: {name}. Available: {list(XAI_REGISTRY)}")
    return XAI_REGISTRY[name](model=model, game=game, **kwargs)
