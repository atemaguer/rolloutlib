"""Opt-in compatibility tests against real FinRL repository checkouts.

Clone FinRL, FinRL-Meta, and FinRL-trading into one directory, then run:

    FINRL_REPOSITORIES_ROOT=/path/to/checkouts \
      uv run --with pandas --with matplotlib --with stable-baselines3 \
      pytest tests/test_finrl_repositories.py -q
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest

from rolloutlib import rollout, wrappers


repositories_root = os.getenv("FINRL_REPOSITORIES_ROOT")
if repositories_root is None:
    pytest.skip(
        "set FINRL_REPOSITORIES_ROOT to sibling FinRL repository checkouts",
        allow_module_level=True,
    )

ROOT = Path(repositories_root)
FINRL = ROOT / "FinRL"
FINRL_META = ROOT / "FinRL-Meta"
FINRL_TRADING = ROOT / "FinRL-trading"

for repository in (FINRL, FINRL_META, FINRL_TRADING):
    if not repository.is_dir():
        pytest.skip(
            f"missing repository checkout: {repository}",
            allow_module_level=True,
        )


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prices(days: int = 12) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array(
        [
            [100.0 + day, 80.0 + 0.5 * day, 120.0 - 0.25 * day]
            for day in range(days)
        ],
        dtype=np.float32,
    )


def test_finrl_stock_environment_accepts_json_tool_actions() -> None:
    module = _load_module(
        "rolloutlib_test_finrl_stock",
        FINRL
        / "finrl"
        / "meta"
        / "env_stock_trading"
        / "env_stocktrading_np.py",
    )
    native = module.StockTradingEnv(
        {
            "price_array": _prices(),
            "tech_array": np.zeros((12, 2), dtype=np.float32),
            "turbulence_array": np.zeros(12, dtype=np.float32),
            "if_train": False,
        }
    )
    env = wrappers.wrap_language_env(
        native,
        instructions="Manage a synthetic three-asset portfolio.",
        tool_name="rebalance",
        argument_name="signals",
        tool_description="Choose one signal per asset from -1 sell to 1 buy.",
        history=12,
    )

    trajectory = rollout(
        env,
        lambda _: {
            "name": "rebalance",
            "arguments": {"signals": [0.2, 0.0, -0.2]},
        },
        max_steps=5,
    )

    assert len(trajectory) == 5
    assert native.day == 5
    assert native.stocks.dtype == np.float32
    assert native.total_asset != native.initial_total_asset


def test_finrl_meta_market_impact_environment_runs_as_language_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(FINRL_META))
    module = __import__(
        "meta.env_market_impact.envs.env_portfolio_optimization_impact",
        fromlist=["PortfolioOptimizationImpactEnv"],
    )
    native = module.PortfolioOptimizationImpactEnv(
        {
            "date_list": [f"2026-01-{day + 1:02d}" for day in range(12)],
            "price_array": _prices(),
            "tech_array": np.zeros((12, 2), dtype=np.float32),
            "volatility_array": np.full((12, 3), 0.02, dtype=np.float32),
            "volume_array": np.full((12, 3), 1_000_000, dtype=np.float32),
            "tic_list": ["AAA", "BBB", "CCC"],
        }
    )
    env = wrappers.wrap_language_env(
        native,
        instructions="Allocate a synthetic three-asset portfolio.",
        tool_name="allocate",
        argument_name="logits",
        tool_description="Choose logits for cash, AAA, BBB, and CCC.",
        history=12,
    )

    trajectory = rollout(
        env,
        lambda _: {
            "name": "allocate",
            "arguments": {"logits": [0.5, 0.2, 0.2, 0.1]},
        },
        max_steps=5,
    )

    assert len(trajectory) == 5
    assert native.time == 5
    assert native.total_asset != 1_000_000
    assert len(native.impact_model.get_impact_history()) > 0


def test_finrl_trading_portfolio_environment_runs_after_gymnasium_normalization() -> (
    None
):
    pytest.importorskip("matplotlib")
    pytest.importorskip("stable_baselines3")
    pd = pytest.importorskip("pandas")

    trading_source = (
        FINRL_TRADING / "src" / "strategies" / "rl_model.py"
    ).read_text()
    assert (
        "from finrl.meta.env_portfolio_allocation.env_portfolio "
        "import StockPortfolioEnv"
    ) in trading_source

    module = _load_module(
        "rolloutlib_test_finrl_portfolio",
        FINRL
        / "finrl"
        / "meta"
        / "env_portfolio_allocation"
        / "env_portfolio.py",
    )
    rows: list[dict[str, Any]] = []
    for day, prices in enumerate(_prices(days=8)):
        for ticker, close, momentum in zip(
            ("AAA", "BBB", "CCC"),
            prices,
            (0.1, 0.0, -0.1),
            strict=True,
        ):
            rows.append(
                {
                    "day": day,
                    "date": f"2026-01-{day + 1:02d}",
                    "tic": ticker,
                    "close": float(close),
                    "mom": momentum,
                    "cov_list": np.eye(3, dtype=np.float32),
                }
            )
    data = pd.DataFrame(rows).set_index("day")
    native = module.StockPortfolioEnv(
        df=data,
        stock_dim=3,
        hmax=100,
        initial_amount=1_000_000,
        transaction_cost_pct=0.001,
        reward_scaling=1.0,
        state_space=3,
        action_space=3,
        tech_indicator_list=["mom"],
    )

    # This FinRL environment returns float64 observations despite advertising
    # float32. Gymnasium's native wrapper restores the declared contract.
    normalized = gym.wrappers.TransformObservation(
        native,
        lambda observation: np.asarray(observation, dtype=np.float32),
        observation_space=native.observation_space,
    )
    env = wrappers.wrap_language_env(
        cast(Any, normalized),
        instructions="Allocate among AAA, BBB, and CCC.",
        tool_name="allocate",
        argument_name="weights",
        tool_description="Choose one score per asset from 0 to 1.",
        history=12,
    )

    trajectory = rollout(
        env,
        lambda _: {
            "name": "allocate",
            "arguments": {"weights": [0.2, 0.4, 0.4]},
        },
        max_steps=5,
    )

    assert len(trajectory) == 5
    assert native.day == 5
    assert native.portfolio_value != native.initial_amount
