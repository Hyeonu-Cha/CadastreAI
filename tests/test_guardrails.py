"""Unit tests for the financial-product guardrail (Task X.04).

Three layers:

1. **Pure pattern coverage** (`screen_query`): a basket of allow- and
   refuse-cases that lock down the regex behaviour. The allow-list
   includes the README's headline example queries — losing those to
   over-eager guardrails would gut the product.
2. **Logging side-effect** (`log_refusal`): refused decisions emit a
   structured WARNING; allowed decisions emit nothing.
3. **Graph wiring** (`guardrail_screen` node + `_route_after_guardrail`):
   the graph short-circuits refused queries to END with the refusal
   text stamped onto `answer_draft`, and allowed queries continue to
   `classify_query` as normal.
"""
from __future__ import annotations

import logging

import pytest

from src.agent import guardrails as gr

# ---------- pattern coverage ------------------------------------------


REFUSE_CASES: list[tuple[str, gr.GuardrailCategory]] = [
    # Mortgage product picking
    ("Which mortgage should I get?", "mortgage_product"),
    ("What home loan is best for me?", "mortgage_product"),
    ("Recommend a mortgage broker", "mortgage_product"),
    ("Best home loan for a first home buyer in NSW", "mortgage_product"),
    ("Should I fix or float my mortgage?", "mortgage_product"),
    ("Should I refinance to get a lower rate?", "mortgage_product"),
    ("Which lender do you recommend?", "mortgage_product"),
    # Insurance product picking
    ("Which landlord insurance should I buy?", "insurance_product"),
    ("Best life insurance for a 30 year old", "insurance_product"),
    ("Recommend home insurance for an investment property", "insurance_product"),
    ("Do I need income protection insurance?", "insurance_product"),
    # Super / managed funds
    ("Which super fund should I be in?", "super_or_managed_fund"),
    ("Best ETF for property exposure", "super_or_managed_fund"),
    ("Should I start an SMSF to buy property?", "super_or_managed_fund"),
    ("Recommend a managed fund for housing", "super_or_managed_fund"),
    # Specific securities
    ("Should I buy shares in Mirvac?", "specific_security_pick"),
    ("Recommend an ASX code for property exposure", "specific_security_pick"),
    ("Best REIT to buy right now", "specific_security_pick"),
]


ALLOW_CASES = [
    # README headline examples — must survive the guardrail.
    "Is Parramatta a good bet for a family on $1.2M? What's the trend and yield?",
    "Sydney suburbs with highest 3yr capital growth and vacancy under 3%",
    "How does CoreLogic's hedonic index differ from ABS 6432.0 methodologically?",
    "What's the 5-year story on housing supply in NSW that I can cite in a piece?",
    # Informational / numeric — same topics, no product-pick language.
    "What's the average mortgage rate in NSW right now?",
    "How does the First Home Guarantee scheme work?",
    "What's the stamp duty on a $1.2M home in NSW?",
    "What's the rental vacancy rate in Brisbane?",
    "What does the RBA cash rate look like over the last decade?",
    "Explain the First Home Super Saver Scheme",
    # Tricky cases that mention finance words but don't ask for a product
    # recommendation.
    "How do landlord insurance premiums compare across capital cities?",
    "What's the gross rental yield on a $900K unit in Melbourne?",
    "Compare the cost of fixed vs variable home loans in published research",
]


@pytest.mark.parametrize("query,category", REFUSE_CASES)
def test_screen_query_refuses_product_recommendations(query, category):
    decision = gr.screen_query(query)
    assert decision.action == "refuse", f"expected refuse for {query!r}"
    assert decision.category == category
    assert decision.refusal_text and len(decision.refusal_text) > 50
    assert decision.reason


@pytest.mark.parametrize("query", ALLOW_CASES)
def test_screen_query_allows_informational_questions(query):
    decision = gr.screen_query(query)
    assert decision.action == "allow", f"expected allow for {query!r}"
    assert decision.category is None
    assert decision.refusal_text is None


def test_screen_query_handles_empty_input():
    """Empty / whitespace queries default to allow — graph upstream guards."""
    assert gr.screen_query("").action == "allow"
    assert gr.screen_query("   ").action == "allow"


def test_screen_query_returns_frozen_decision():
    """`GuardrailDecision` is frozen — no in-place mutation."""
    import dataclasses

    decision = gr.screen_query("Which mortgage should I get?")
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.action = "allow"  # type: ignore[misc]


def test_refusal_text_offers_redirection():
    """Refusals must point users toward the kind of question we can help with."""
    for category in (
        "mortgage_product",
        "insurance_product",
        "super_or_managed_fund",
        "specific_security_pick",
    ):
        text = gr._REFUSAL_TEMPLATES[category]
        # Names what we won't do.
        assert "can't" in text.lower() or "can not" in text.lower()
        # Names a licensed alternative.
        assert "licensed" in text.lower() or "adviser" in text.lower() or "broker" in text.lower()


# ---------- logging side-effect ---------------------------------------


def test_log_refusal_emits_warning_for_refused(caplog):
    decision = gr.screen_query("Which mortgage should I get?")
    with caplog.at_level(logging.WARNING, logger=gr.log.name):
        gr.log_refusal("Which mortgage should I get?", decision)
    refusal_records = [r for r in caplog.records if "guardrail refusal" in r.getMessage()]
    assert refusal_records, "expected exactly one refusal log line"
    msg = refusal_records[0].getMessage()
    assert "category=mortgage_product" in msg
    assert "Which mortgage should I get?" in msg


def test_log_refusal_silent_for_allow(caplog):
    decision = gr.screen_query("What's the cash rate today?")
    with caplog.at_level(logging.WARNING, logger=gr.log.name):
        gr.log_refusal("What's the cash rate today?", decision)
    assert not [r for r in caplog.records if "guardrail" in r.getMessage()]


# ---------- graph node wiring -----------------------------------------


def test_guardrail_screen_node_passes_through_safe_query():
    """Allow path: node returns an empty patch and graph continues."""
    from src.agent import nodes

    state = {
        "messages": [{"role": "user", "content": "What's the cash rate today?"}],
        "iteration_count": 0,
    }
    out = nodes.guardrail_screen(state)
    assert out == {}, "expected no state changes on allow"


def test_guardrail_screen_node_short_circuits_blocked_query():
    """Refuse path: node stamps refusal text + reflection.is_complete."""
    from src.agent import nodes

    state = {
        "messages": [{"role": "user", "content": "Which mortgage should I get?"}],
        "iteration_count": 0,
    }
    out = nodes.guardrail_screen(state)
    assert "answer_draft" in out
    assert "can't recommend specific mortgage" in out["answer_draft"]
    # Marks refusal so UI can label it distinctly.
    assert out["guardrail"]["blocked"] is True
    assert out["guardrail"]["category"] == "mortgage_product"
    # Reflection is sealed so the graph terminates immediately.
    assert out["reflection"]["is_complete"] is True
    # Assistant message is appended to the conversation log.
    assert any(
        m.get("role") == "assistant" and "can't recommend" in m["content"]
        for m in out["messages"]
    )


def test_route_after_guardrail():
    from src.agent import graph as graph_mod

    blocked_state = {"guardrail": {"blocked": True, "category": "mortgage_product"}}
    assert graph_mod._route_after_guardrail(blocked_state) == "end"

    allowed_state = {"guardrail": {}}  # default branch
    assert graph_mod._route_after_guardrail(allowed_state) == "continue"

    no_guardrail_state: dict = {}
    assert graph_mod._route_after_guardrail(no_guardrail_state) == "continue"


def test_graph_short_circuits_on_refusal(monkeypatch):
    """End-to-end: a financial-product query never reaches the classifier."""
    from src.agent import nodes
    from src.agent.graph import build_graph, initial_state

    classify_called = {"n": 0}

    def stub_classifier(_state):
        classify_called["n"] += 1
        return {"classification": {}, "query_type": "factual"}

    monkeypatch.setattr(nodes, "classify_query", stub_classifier)
    g = build_graph()
    out = g.invoke(initial_state("Which mortgage should I get?"))

    assert classify_called["n"] == 0, "classifier must not run on a blocked query"
    assert "can't recommend specific mortgage" in out["answer_draft"]
    assert out["guardrail"]["blocked"] is True


def test_graph_passes_safe_query_to_classifier(monkeypatch):
    """End-to-end: a safe query proceeds through guardrail to classifier."""
    from src.agent import nodes
    from src.agent.graph import build_graph, initial_state

    classify_called = {"n": 0}

    def stub_classifier(_state):
        classify_called["n"] += 1
        return {
            "classification": {
                "persona": "general",
                "query_type": "factual",
                "needs_docs": True,
                "needs_data": False,
                "needs_decomposition": False,
            },
            "query_type": "factual",
        }

    monkeypatch.setattr(nodes, "classify_query", stub_classifier)
    monkeypatch.setattr(
        nodes,
        "_plan_subquestion",
        lambda q, persona=None: {"use_docs": False, "tool_calls": []},
    )
    monkeypatch.setattr(
        nodes,
        "synthesize",
        lambda state: {
            "answer_draft": "ok",
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": "ok"}],
        },
    )
    monkeypatch.setattr(
        nodes,
        "reflect",
        lambda _state: {"reflection": {"is_complete": True, "missing": [], "refined_query": None}},
    )
    g = build_graph()
    out = g.invoke(initial_state("What's the average mortgage rate in NSW?"))

    assert classify_called["n"] == 1
    assert out.get("guardrail", {}).get("blocked", False) is False
    assert out["answer_draft"] == "ok"
