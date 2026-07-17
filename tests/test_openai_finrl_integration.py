"""Opt-in live OpenAI policy test against FinRL's Gymnasium stock environment.

Run with ``RUN_OPENAI_FINRL_INTEGRATION=1``, ``FINRL_REPOSITORIES_ROOT``, and
``OPENAI_API_KEY``. The test uses synthetic prices and never places real trades.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest


if os.getenv("RUN_OPENAI_FINRL_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_OPENAI_FINRL_INTEGRATION=1 to run the paid OpenAI FinRL test",
        allow_module_level=True,
    )

repositories_root = os.getenv("FINRL_REPOSITORIES_ROOT")
if repositories_root is None:
    pytest.skip(
        "set FINRL_REPOSITORIES_ROOT to sibling FinRL repository checkouts",
        allow_module_level=True,
    )

finrl_source = (
    Path(repositories_root)
    / "FinRL"
    / "finrl"
    / "meta"
    / "env_stock_trading"
    / "env_stocktrading_np.py"
)
if not finrl_source.is_file():
    pytest.skip(f"missing FinRL environment: {finrl_source}", allow_module_level=True)

openai = pytest.importorskip("openai")

from rolloutlib import rollout, wrappers  # noqa: E402
from rolloutlib.policies import OpenAIResponsesPolicy  # noqa: E402


def test_openai_agent_manages_a_finrl_portfolio_for_five_steps() -> None:
    spec = importlib.util.spec_from_file_location(
        "rolloutlib_test_openai_finrl",
        finrl_source,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tickers = ("AAA", "BBB", "CCC")
    prices = np.array(
        [
            [100.0 + day, 80.0 + 0.5 * day, 120.0 - 0.25 * day]
            for day in range(12)
        ],
        dtype=np.float32,
    )
    native = module.StockTradingEnv(
        {
            "price_array": prices,
            "tech_array": np.zeros((12, 2), dtype=np.float32),
            "turbulence_array": np.zeros(12, dtype=np.float32),
            "if_train": False,
        }
    )

    def market_state(_: np.ndarray[Any, np.dtype[np.float32]]) -> dict[str, Any]:
        day = cast(int, native.day)
        holdings = cast(np.ndarray[Any, Any], native.stocks)
        return {
            "day": day,
            "cash": round(float(native.amount), 2),
            "portfolio_value": round(float(native.total_asset), 2),
            "prices": {
                ticker: round(float(price), 2)
                for ticker, price in zip(tickers, native.price_ary[day], strict=True)
            },
            "holdings": {
                ticker: int(shares)
                for ticker, shares in zip(tickers, holdings, strict=True)
            },
            "signal_meaning": {
                "-1": "sell up to 100 shares",
                "0": "hold",
                "1": "buy up to 100 shares",
            },
        }

    env = wrappers.wrap_language_env(
        native,
        state=market_state,
        instructions=(
            "This is a synthetic backtest, not live trading. Manage AAA, BBB, "
            "and CCC to increase portfolio value while avoiding unnecessary "
            "turnover. Review the full episode history before acting."
        ),
        tool_name="rebalance",
        argument_name="signals",
        tool_description=(
            "Choose exactly three signals ordered as AAA, BBB, CCC. "
            "Each signal must be between -1 and 1."
        ),
        history=16,
    )
    model = os.getenv("OPENAI_FINRL_MODEL", "gpt-5.6-sol")
    policy = OpenAIResponsesPolicy.from_env(
        env,
        client=openai.OpenAI(),
        model=model,
        reasoning={"effort": "none"},
        instructions=(
            "Use the market state and prior actions to choose one valid "
            "rebalance call. Call the tool exactly once."
        ),
        max_output_tokens=512,
        store=False,
    )

    try:
        trajectory = rollout(
            env,
            policy,
            seed=7,
            max_steps=5,
            metadata={
                "model": model,
                "environment": "FinRL StockTradingEnv",
                "market": "synthetic",
            },
        )
    finally:
        env.close()

    assert len(trajectory) == 5
    assert native.day == 5
    for step in trajectory:
        assert step.action["name"] == "rebalance"
        signals = np.asarray(step.action["arguments"]["signals"])
        assert signals.shape == (3,)
        assert np.issubdtype(signals.dtype, np.number)
        assert np.all((-1 <= signals) & (signals <= 1))
        assert step.policy_info["response_id"].startswith("resp_")
        assert "gpt-5.6" in step.policy_info["model"]
        assert step.policy_info["reasoning_effort"] == "none"

    print(
        "OpenAI FinRL validation: "
        f"model={model}, reasoning=none, steps={len(trajectory)}, "
        f"actions={trajectory.actions}, "
        f"final_value={float(native.total_asset):.2f}"
    )
