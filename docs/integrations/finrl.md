# FinRL trading agents

Rolloutlib can present continuous FinRL portfolio environments directly to a
language policy. The integration uses the existing Gymnasium action and
observation spaces; it does not introduce a FinRL-specific environment base
class.

This is best treated as a research and backtesting workflow. Wrapping an
environment does not add order limits, approval gates, market-data validation,
or other controls needed for live trading.

## Compatibility across the FinRL repositories

| Repository | Environment path tested | Result |
| --- | --- | --- |
| FinRL | `env_stocktrading_np.StockTradingEnv` | Direct Gymnasium integration |
| FinRL-Meta | `PortfolioOptimizationImpactEnv` | Direct Gymnasium integration |
| FinRL-trading | FinRL's `StockPortfolioEnv`, imported by `rl_model.py` | Works after a standard Gymnasium observation-dtype wrapper |

FinRL-trading is a trading platform rather than a separate environment
collection. Its DRL strategy imports `StockPortfolioEnv` from FinRL, so the
same language wrapper can sit around that environment before rollout.

FinRL-Meta contains both current Gymnasium environments and older OpenAI Gym
environments. Prefer a current Gymnasium environment when possible. A legacy
environment whose `reset` returns only an observation or whose `step` returns
four values must first go through the Gymnasium/Shimmy compatibility path.

## Wrapping a stock environment

The native action remains FinRL's `Box` vector. Rolloutlib exposes that vector
as a strictly validated function argument and converts the model's JSON array
back to the NumPy array expected by FinRL.

```python
from openai import OpenAI

from rolloutlib import rollout, wrappers
from rolloutlib.policies import OpenAIResponsesPolicy


# Construct this with FinRL as usual.
native = StockTradingEnv(config)
tickers = ("AAA", "BBB", "CCC")


def market_state(_observation):
    day = native.day
    return {
        "day": day,
        "cash": round(float(native.amount), 2),
        "portfolio_value": round(float(native.total_asset), 2),
        "prices": {
            ticker: float(price)
            for ticker, price in zip(
                tickers, native.price_ary[day], strict=True
            )
        },
        "holdings": {
            ticker: int(shares)
            for ticker, shares in zip(tickers, native.stocks, strict=True)
        },
    }


env = wrappers.wrap_language_env(
    native,
    state=market_state,
    instructions=(
        "This is a synthetic backtest. Manage AAA, BBB, and CCC while "
        "avoiding unnecessary turnover."
    ),
    tool_name="rebalance",
    argument_name="signals",
    tool_description=(
        "Choose exactly three signals ordered as AAA, BBB, CCC. "
        "Each signal must be between -1 and 1."
    ),
    history=32,
)

policy = OpenAIResponsesPolicy.from_env(
    env,
    client=OpenAI(),
    model="gpt-5.6-sol",
    reasoning={"effort": "none"},
    instructions="Review the episode history and call rebalance exactly once.",
    store=False,
)

trajectory = rollout(env, policy, max_steps=20)
```

The `state` function is the only domain-specific translation in this example.
It gives the model meaningful field names instead of an unlabeled numerical
state vector. The environment still owns transitions, rewards, portfolio
accounting, and action bounds.

`history` records the model's prior allocations and subsequent market states.
The recorded tool calls use JSON arrays while FinRL receives native NumPy
arrays.

## Correcting an environment's declared observation contract

The portfolio environment used by FinRL-trading currently constructs
`float64` observations while declaring a `float32` observation space. Compose
Gymnasium's standard wrapper before the language wrappers:

```python
import gymnasium as gym
import numpy as np

native = StockPortfolioEnv(...)
native = gym.wrappers.TransformObservation(
    native,
    lambda observation: np.asarray(observation, dtype=np.float32),
    observation_space=native.observation_space,
)
env = wrappers.wrap_language_env(native, ...)
```

This keeps environment contract repair in Gymnasium terminology and keeps
rolloutlib focused on the language-facing transformation.

## Reproducing the repository tests

Clone the repositories as siblings:

```console
mkdir finrl-repositories
git clone https://github.com/AI4Finance-Foundation/FinRL.git finrl-repositories/FinRL
git clone https://github.com/AI4Finance-Foundation/FinRL-Meta.git finrl-repositories/FinRL-Meta
git clone https://github.com/AI4Finance-Foundation/FinRL-trading.git finrl-repositories/FinRL-trading
```

Run deterministic five-step trajectories against all three:

```console
FINRL_REPOSITORIES_ROOT="$PWD/finrl-repositories" \
  uv run --with pandas --with matplotlib --with stable-baselines3 \
  pytest tests/test_finrl_repositories.py -q
```

The opt-in live test makes five paid OpenAI Responses API requests against a
synthetic FinRL market:

```console
export OPENAI_API_KEY=...
RUN_OPENAI_FINRL_INTEGRATION=1 \
FINRL_REPOSITORIES_ROOT="$PWD/finrl-repositories" \
  uv run --extra openai \
  pytest tests/test_openai_finrl_integration.py -q -s
```

Set `OPENAI_FINRL_MODEL` to override the default model. No broker is configured
and no real order can be placed by either test.

## Validating against historical market data

The historical integration test downloads adjusted daily closes for AAPL,
MSFT, and JPM from Yahoo Finance for the fixed period from January 2 through
February 28, 2025. It derives one-day returns, trailing five-day momentum, and
trailing volatility, then runs ten paid model decisions through FinRL's real
`StockTradingEnv`:

```console
export OPENAI_API_KEY=...
RUN_OPENAI_FINRL_HISTORICAL_INTEGRATION=1 \
FINRL_REPOSITORIES_ROOT="$PWD/finrl-repositories" \
  uv run --extra openai-finrl \
  pytest tests/test_openai_finrl_historical_integration.py -q -s
```

The observation for each decision contains only that date's adjusted close,
backward-looking features, cash, holdings, and prior episode history. Future
rows are never exposed to the language policy. The test downloads data at run
time, so it is opt-in alongside the paid model calls and will skip during the
default offline test suite.

This is an infrastructure backtest, not a performance benchmark or trading
recommendation. It has no broker connection and cannot place real orders.

### Auditing a reasoning-enabled run

OpenAI does not expose raw reasoning tokens. For an auditable trajectory, the
test can request an API reasoning summary and use a standard Gymnasium
`ActionWrapper` to add a bounded public `rationale` string to the native action.
The wrapper removes that string before passing the unchanged signal vector to
FinRL:

```console
RUN_OPENAI_FINRL_HISTORICAL_INTEGRATION=1 \
FINRL_REPOSITORIES_ROOT="$PWD/finrl-repositories" \
OPENAI_FINRL_MODEL=gpt-5.6-sol \
OPENAI_FINRL_REASONING_EFFORT=high \
OPENAI_FINRL_REASONING_MODE=pro \
OPENAI_FINRL_REASONING_SUMMARY=auto \
OPENAI_FINRL_AUDIT_RATIONALE=1 \
OPENAI_FINRL_MAX_OUTPUT_TOKENS=4096 \
  uv run --extra openai-finrl \
  pytest tests/test_openai_finrl_historical_integration.py -q -s
```

When the API returns a summary, `OpenAIResponsesPolicy` records it as
`step.policy_info["reasoning_summary"]`. The public rationale is part of the
tool call and therefore remains in `step.action` and the episode history.
Summaries are optional API output, while the audited rationale is required by
the action space on every step.
