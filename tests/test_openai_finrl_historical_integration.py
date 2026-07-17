"""Opt-in OpenAI/FinRL backtest using real historical Yahoo Finance data.

The test downloads adjusted daily prices for AAPL, MSFT, and JPM from
2025-01-02 through 2025-02-28, then lets the model trade the first ten
transitions. It never connects to a broker or submits real orders.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest


if os.getenv("RUN_OPENAI_FINRL_HISTORICAL_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_OPENAI_FINRL_HISTORICAL_INTEGRATION=1 to run the paid test",
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
yf = pytest.importorskip("yfinance")

from rolloutlib import rollout, spaces, wrappers  # noqa: E402
from rolloutlib.policies import OpenAIResponsesPolicy  # noqa: E402


class PublicRationaleActionWrapper(
    gym.ActionWrapper[Any, dict[str, Any], np.ndarray[Any, Any]]
):
    """Attach an auditable rationale without changing FinRL's native action."""

    def __init__(self, env: gym.Env[Any, np.ndarray[Any, Any]]) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Dict(
            {
                "signals": env.action_space,
                "rationale": spaces.TextSpace(
                    min_length=1,
                    max_length=800,
                ),
            }
        )

    def action(self, action: dict[str, Any]) -> np.ndarray[Any, Any]:
        return cast(np.ndarray[Any, Any], action["signals"])


def test_openai_agent_trades_ten_days_of_real_historical_prices() -> None:
    tickers = ("AAPL", "MSFT", "JPM")
    history = yf.download(
        list(tickers),
        start="2025-01-02",
        end="2025-03-01",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
        timeout=20,
    )
    assert not history.empty
    close = history["Close"].reindex(columns=tickers).dropna()
    assert len(close) >= 11
    assert not close.isna().any().any()

    daily_returns = close.pct_change(fill_method=None).fillna(0.0)
    five_day_momentum = (close / close.shift(5) - 1.0).fillna(0.0)
    rolling_volatility = (
        daily_returns.rolling(5).std().mean(axis=1).fillna(0.0) * 1_000
    )
    technicals = np.concatenate(
        (
            daily_returns.to_numpy(dtype=np.float32),
            five_day_momentum.to_numpy(dtype=np.float32),
        ),
        axis=1,
    )

    spec = importlib.util.spec_from_file_location(
        "rolloutlib_test_openai_finrl_historical",
        finrl_source,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.StockTradingEnv(
        {
            "price_array": close.to_numpy(dtype=np.float32),
            "tech_array": technicals,
            "turbulence_array": rolling_volatility.to_numpy(dtype=np.float32),
            "if_train": False,
        }
    )
    dates = tuple(timestamp.date().isoformat() for timestamp in close.index)

    def market_state(_: np.ndarray[Any, np.dtype[np.float32]]) -> dict[str, Any]:
        day = cast(int, native.day)
        holdings = cast(np.ndarray[Any, Any], native.stocks)
        return {
            "date": dates[day],
            "cash": round(float(native.amount), 2),
            "portfolio_value": round(float(native.total_asset), 2),
            "adjusted_close": {
                ticker: round(float(price), 2)
                for ticker, price in zip(tickers, native.price_ary[day], strict=True)
            },
            "one_day_return_pct": {
                ticker: round(float(value * 100), 3)
                for ticker, value in zip(
                    tickers,
                    daily_returns.iloc[day],
                    strict=True,
                )
            },
            "five_day_momentum_pct": {
                ticker: round(float(value * 100), 3)
                for ticker, value in zip(
                    tickers,
                    five_day_momentum.iloc[day],
                    strict=True,
                )
            },
            "holdings": {
                ticker: int(shares)
                for ticker, shares in zip(tickers, holdings, strict=True)
            },
            "signal_order": list(tickers),
            "signal_meaning": {
                "-1": "sell up to 100 shares",
                "0": "hold",
                "1": "buy up to 100 shares",
            },
        }

    audit_rationale = os.getenv("OPENAI_FINRL_AUDIT_RATIONALE") == "1"
    action_env = PublicRationaleActionWrapper(native) if audit_rationale else native
    argument_name = "decision" if audit_rationale else "signals"
    env = wrappers.wrap_language_env(
        action_env,
        state=market_state,
        instructions=(
            "This is an offline historical backtest with adjusted daily prices. "
            "Only use information in the current observation and episode history; "
            "do not assume knowledge of future prices. Manage AAPL, MSFT, and JPM "
            "to grow portfolio value while limiting unnecessary turnover."
        ),
        tool_name="rebalance",
        argument_name=argument_name,
        tool_description=(
            (
                "Choose exactly three signals ordered as AAPL, MSFT, JPM and "
                "provide a concise public rationale grounded in the supplied "
                "market state. "
            )
            if audit_rationale
            else "Choose exactly three signals ordered as AAPL, MSFT, JPM. "
        )
        + (
            "Each signal must be between -1 and 1; magnitudes at or below "
            "0.1 are treated as holds by this FinRL environment."
        ),
        history=24,
    )
    model = os.getenv("OPENAI_FINRL_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.getenv("OPENAI_FINRL_REASONING_EFFORT", "none")
    reasoning_summary = os.getenv("OPENAI_FINRL_REASONING_SUMMARY")
    reasoning: dict[str, str] = {"effort": reasoning_effort}
    if reasoning_summary is not None:
        reasoning["summary"] = reasoning_summary
    reasoning_mode = os.getenv("OPENAI_FINRL_REASONING_MODE")
    if reasoning_mode is not None:
        reasoning["mode"] = reasoning_mode
    policy = OpenAIResponsesPolicy.from_env(
        env,
        client=openai.OpenAI(),
        model=model,
        reasoning=reasoning,
        instructions=(
            "Review prices, returns, momentum, holdings, cash, and prior actions. "
            + (
                "Write a concise, auditable rationale of at most three sentences. "
                if audit_rationale
                else ""
            )
            + "Call rebalance exactly once with a valid signal vector."
        ),
        max_output_tokens=int(os.getenv("OPENAI_FINRL_MAX_OUTPUT_TOKENS", "512")),
        store=False,
    )

    initial_value = float(native.initial_capital)
    try:
        trajectory = rollout(
            env,
            policy,
            seed=7,
            max_steps=10,
            metadata={
                "model": model,
                "environment": "FinRL StockTradingEnv",
                "data_source": "Yahoo Finance",
                "tickers": tickers,
                "start": dates[0],
            },
        )
    finally:
        env.close()

    final_value = float(native.total_asset)
    actions: list[list[float]] = []
    reasoning_summaries: list[str | None] = []
    public_rationales: list[str] = []
    assert len(trajectory) == 10
    assert native.day == 10
    assert dates[native.day] == "2025-01-17"
    for step in trajectory:
        argument = step.action["arguments"][argument_name]
        if audit_rationale:
            decision = cast(dict[str, Any], argument)
            signals = np.asarray(decision["signals"], dtype=float)
            public_rationales.append(cast(str, decision["rationale"]))
        else:
            signals = np.asarray(argument, dtype=float)
        assert signals.shape == (3,)
        assert np.all((-1 <= signals) & (signals <= 1))
        assert step.policy_info["response_id"].startswith("resp_")
        assert "gpt-5.6" in step.policy_info["model"]
        assert step.policy_info["reasoning_effort"] == reasoning_effort
        summary = step.policy_info.get("reasoning_summary")
        reasoning_summaries.append(summary if isinstance(summary, str) else None)
        actions.append(signals.round(3).tolist())

    if os.getenv("OPENAI_FINRL_REQUIRE_REASONING_SUMMARY") == "1":
        assert all(reasoning_summaries)
    if audit_rationale:
        assert len(public_rationales) == len(trajectory)
    print(
        "OpenAI FinRL historical validation: "
        f"model={model}, period={dates[0]}..{dates[native.day]}, "
        f"steps={len(trajectory)}, actions={actions}, "
        f"initial_value={initial_value:.2f}, final_value={final_value:.2f}, "
        f"return_pct={(final_value / initial_value - 1) * 100:.4f}"
    )
    for index, summary in enumerate(reasoning_summaries, start=1):
        print(f"Step {index} reasoning summary: {summary or '<not returned>'}")
    for index, rationale in enumerate(public_rationales, start=1):
        print(f"Step {index} public rationale: {rationale}")
