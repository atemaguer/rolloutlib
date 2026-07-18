# Getting started

Rolloutlib provides standard, composable APIs for agentic RL post-training.
Environments define interactive tasks, language-model policies choose actions,
rollouts record training trajectories, and graders produce reward and
evaluation signals.

## Requirements

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

## Install from PyPI

```console
pip install rolloutlib
```

## Install from source

```console
git clone https://github.com/atemaguer/rolloutlib.git
cd rolloutlib
uv sync
```

## A minimal rollout

An environment owns task state and reward semantics. A policy owns model
sampling. The rollout function connects them and records the episode:

```python
import gymnasium as gym

from rolloutlib import Env, rollout


class AnswerEnv(Env):
    action_space = gym.spaces.Text(min_length=1, max_length=100)
    observation_space = gym.spaces.Text(min_length=1, max_length=100)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return "What is 6 × 7?", {}

    def step(self, action):
        correct = action.strip() == "42"
        return "done", float(correct), True, False, {}


trajectory = rollout(AnswerEnv(), lambda observation: "42")
assert trajectory.total_reward == 1.0
```

The policy can instead return `PolicyOutput` to retain tokens, log
probabilities, and sampling metadata. See [Policies and
rollouts](concepts/rollouts.md).

For the optional benchmark dataset loaders:

```console
uv sync --extra benchmarks
```

## Verify the checkout

```console
uv run pytest -q
uv run ruff check rolloutlib tests
uv run pyright
uv build
```

Build the documentation locally:

```console
uv sync --group docs
uv run --group docs mkdocs build --strict
```

The generated site is written to `site/`. Serve it locally with:

```console
uv run --group docs mkdocs serve
```

## OpenAI chess integration

The opt-in chess test wraps the external `BulletChess-v0` Gymnasium environment
for a language agent. It runs 20 self-play steps with `gpt-5.6-luna` at
reasoning effort `none`, sending each board image and structured state to the
OpenAI Responses API and applying every selected legal move:

```console
uv sync --extra openai-chess
export OPENAI_API_KEY=...
RUN_OPENAI_CHESS_INTEGRATION=1 uv run pytest \
  tests/test_openai_chess_integration.py -q
```

The default model can be changed with `OPENAI_CHESS_MODEL`. The test makes 20
paid API requests and is skipped unless `RUN_OPENAI_CHESS_INTEGRATION=1`.

## Tinker integration

The Tinker smoke and AIME parity tests are opt-in. Install the Tinker SDK,
Tinker Cookbook, and benchmark datasets, configure your Tinker credentials,
and run:

```console
uv run pip install tinker tinker-cookbook datasets
export RUN_TINKER_INTEGRATION=1
uv run pytest tests/test_tinker_policy.py tests/test_tinker_aime.py -q
```

Useful overrides include `TINKER_MODEL_NAME`, `TINKER_MODEL_PATH`,
`TINKER_RENDERER`, `TINKER_MAX_TOKENS`, and `TINKER_AIME_LIMIT`. These tests
make model/API requests and may incur provider costs.
