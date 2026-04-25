"""Pretty-print extracted logical formulas and rules."""
from __future__ import annotations
from typing import Any


def render_rules(rules: list[str], title: str = "Extracted Rules") -> str:
    """Format a list of rule strings into a human-readable block."""
    lines = [f"=== {title} ==="]
    for i, rule in enumerate(rules, 1):
        lines.append(f"{i}. {rule}")
    return "\n".join(lines)


def render_sympy_formula(formula: Any) -> str:
    """Convert a sympy expression to a readable string."""
    try:
        from sympy import pretty
        return pretty(formula, use_unicode=True)
    except Exception:
        return str(formula)


def render_explanation(
    explanation: dict,
    game_name: str = "",
    xai_name: str = "",
    game: Any = None,
) -> str:
    """
    Render a full explanation dict to a human-readable string.

    When ``game`` is provided and ``explanation`` contains ``sympy_formulas``,
    appends a "Natural-Language Rules" section with canonical template matching.
    Existing ``### Rules:`` and ``### Symbolic Formulas:`` sections are unchanged.
    """
    lines = []
    if game_name or xai_name:
        lines.append(f"## Explanation: {game_name} / {xai_name}")
    lines.append(f"Method: {explanation.get('method', 'unknown')}")
    lines.append(f"Summary: {explanation.get('summary', '')}")

    if "rules" in explanation:
        lines.append("\n### Rules:")
        for rule in explanation["rules"]:
            lines.append(str(rule)[:200])

    if "sympy_formulas" in explanation and explanation["sympy_formulas"]:
        lines.append("\n### Symbolic Formulas:")
        for f in explanation["sympy_formulas"][:5]:
            lines.append(f"  {render_sympy_formula(f)}")

    if "concept_scores" in explanation and explanation["concept_scores"]:
        lines.append("\n### Concept Probe Scores (top 10):")
        sorted_concepts = sorted(explanation["concept_scores"].items(), key=lambda x: -x[1])[:10]
        for name, score in sorted_concepts:
            lines.append(f"  {name}: {score:.3f}")

    if "poly_expression" in explanation:
        lines.append(f"\n### Polynomial Expression:\n  {explanation['poly_expression'][:200]}")

    # Natural-language rules — only when game is provided and formulas exist
    if game is not None and explanation.get("sympy_formulas"):
        try:
            from .templates import render_clauses_as_nl
            nl = render_clauses_as_nl(explanation["sympy_formulas"], game)
            matched, total = nl["matched"], nl["total"]
            rate = f"{matched}/{total} = {nl['canonical_match_rate']:.2f}"
            lines.append(f"\n### Natural-Language Rules (canonical match rate: {rate}):")
            for i, text in enumerate(nl["nl_rules"], 1):
                rule_obj = None
                try:
                    from .templates import _formula_to_atoms, match_clause
                    atoms = _formula_to_atoms(explanation["sympy_formulas"][i - 1], game)
                    rule_obj = match_clause(atoms, game) if atoms is not None else None
                except Exception:
                    pass
                if rule_obj is not None:
                    lines.append(f"{i}. ✓ [{rule_obj.template}] {text}")
                else:
                    formula_str = render_sympy_formula(explanation["sympy_formulas"][i - 1])
                    lines.append(f"{i}. ✗ (non-canonical)  {formula_str}")

            lines.append("\n### By template:")
            for name, count in sorted(nl["by_template"].items()):
                lines.append(f"  {name:<22} {count}")
        except Exception as exc:
            lines.append(f"\n### Natural-Language Rules: (unavailable: {exc})")

    return "\n".join(lines)
