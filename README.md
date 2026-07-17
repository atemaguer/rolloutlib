# rolloutlib

Gymnasium-style environments and rollout primitives for agentic RL
post-training.

Rolloutlib keeps the small, composable contracts that make Gymnasium and Tinker
useful: environments expose `reset`/`step` and spaces; rollout code records
interactions; RL training systems can consume those records through their own
backend-specific data adapters. Rolloutlib does not own model sampling.

## Install

```bash
pip install rolloutlib
```

## Environments

Synchronous environments are real `gymnasium.Env` instances:

```python
import gymnasium as gym
from rolloutlib import Env


class EchoEnv(Env):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Text(min_length=0, max_length=20)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return "ready", {}

    def step(self, action):
        return str(action), float(action), True, False, {}
```

Async environments preserve the same value-level contract:

```python
from rolloutlib import AsyncEnv


class ToolEnv(AsyncEnv):
    action_space = ...
    observation_space = ...

    async def reset(self, *, seed=None, options=None):
        await super().reset(seed=seed, options=options)
        return observation, {}

    async def step(self, action):
        result = await execute_tool(action)
        return next_observation, reward, terminated, truncated, {"result": result}
```

Use `as_async` and `as_sync` when integrating an existing environment with the
other calling convention. Calls on one environment instance are serialized;
concurrency belongs across independent instances.

Grading that determines reward belongs inside `step`. Single-turn environments
may return a `Score` directly from `evaluate`; existing environments can be
composed with a grading wrapper:

```python
from rolloutlib import GradingWrapper

environment = GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=lambda env, action: (item, env.state, action),
)
```

`AsyncGradingWrapper` follows the same contract and awaits asynchronous graders.
By default, wrappers grade terminal or truncated steps, replace the scalar
reward with `Score.value`, and preserve the complete score under `info["score"]`.

## Spaces

Rolloutlib accepts every Gymnasium space and supplies common spaces for text,
tokens, messages, and tool calls:

```python
import gymnasium as gym
from rolloutlib import spaces

token_sequence = spaces.tokens.sequence(vocab_size=128_000)
chat = spaces.messages.chat(min_length=1)
tool_call = spaces.tools.call(
    {
        "search": gym.spaces.Dict(
            {"query": spaces.text.text(min_length=1, max_length=1_000)}
        )
    }
)
tool_calls = spaces.tools.calls({"search": gym.spaces.Dict({"query": spaces.text.text()})})
```

Structured values are ordinary dictionaries and lists. Pydantic validates them
at space boundaries; applications do not need to construct framework-specific
message or tool-call model objects. `TextSpace` accepts all Unicode strings
that satisfy its length constraints. Its `sample_alphabet` only controls
random sampling.

## Rollouts

The rollout layer records environment interactions while leaving sampling to
user code. `Policy` is the synchronous callable contract and `AsyncPolicy` is
its async-compatible counterpart. Either may return a raw action or a
`PolicyOutput` containing model-side information such as generated tokens and
behavior-policy log probabilities.

```python
from rolloutlib import PolicyOutput, rollout


def policy(observation):
    tokens, text = model.generate(observation)
    return PolicyOutput(
        action=text,
        tokens=tokens,
        logprobs=model.logprobs,
    )


trajectory = rollout(environment, policy)
```

The stable data records are `Step`, `Trajectory`, and `TrajectoryGroup`:

```python
from rolloutlib.rollouts import rollout_group

group = rollout_group(
    item,
    make_env,
    policy,
    num_rollouts=8,
    item_id="problem-17",
)

group.trajectories  # independent episodes for one item
group.rewards       # scalar scores consumed by an algorithm
```

`Step.info` belongs to the environment; `Step.policy_info` belongs to the
sampling policy. `Step.policy_tokens`, `Step.policy_logprobs`, and
`Step.policy_stop_reason` preserve common sampling fields without coupling the
core to a model backend. `terminated` and `truncated` retain Gymnasium’s distinction.
`rollout` does not close its environment. `rollout_group` owns and closes the
fresh environments it creates. `arollout` and `arollout_group` provide async
counterparts with bounded concurrency.

Backend-specific policies are ordinary user code. For example, a Tinker
policy can build a generation prompt with a Tinker renderer, call
`SamplingClient.sample` (or `sample_async`), parse the returned tokens, and
return `PolicyOutput`. This keeps Tinker optional while preserving the same
rollout contract.

## Datasets

Datasets are sources of pre-rollout work, not collections of completed
trajectories:

```python
from rolloutlib.datasets import Dataset, RLDataset

items = Dataset([problem_a, problem_b], metadata={"split": "train"})
rl_items = RLDataset(
    [problem_a, problem_b],
    make_env=lambda problem: ProblemEnv(problem),
    get_item_id=lambda problem: problem.id,
)
```

Batching, shuffling, repetition, and the number of rollouts per item remain
training-loop concerns. Larger systems can provide their own sequence or
streaming dataset without inheriting from these convenience containers.

## Graders

`Grader` defines one operation: `grade(input) -> Score`. `AsyncGrader` provides
the corresponding awaitable operation. Every grader has an `input_space`
describing its complete input, so it can safely score a response, action,
trajectory, tool trace, or richer application-defined record.

Rolloutlib provides three grader families:

- `RubricGrader` applies a portable, human-defined rubric through a user-owned
  judge, commonly an LLM judge.
- `RewardGrader` evaluates named programmatic reward functions.
- `CompositeGrader` combines complete graders while preserving their nested
  results.

Each has an asynchronous counterpart.

```python
from rolloutlib import spaces
from rolloutlib.graders import (
    AsyncCompositeGrader,
    AsyncRubricGrader,
    Criterion,
    Level,
    RewardGrader,
    Rubric,
    Score,
)

rubric = Rubric(
    id="answer-quality",
    version="1",
    criteria=(
        Criterion(
            id="correctness",
            description="The answer is correct.",
            weight=4.0,
            levels=(
                Level(
                    id="correct",
                    description="The answer and reasoning are correct.",
                    score=1.0,
                ),
                Level(
                    id="partial",
                    description="The answer is correct with incomplete reasoning.",
                    score=0.5,
                ),
                Level(
                    id="incorrect",
                    description="The answer is incorrect.",
                    score=0.0,
                ),
            ),
        ),
        Criterion(
            id="format",
            description="The requested format is followed.",
            weight=1.0,
        ),
    ),
    instructions="Grade only the submitted answer.",
)


async def judge(answer: str, rubric: Rubric):
    request = render_judge_request(answer, rubric)
    response = await call_judge_model(request)
    return {
        "correctness": Score(
            response.correctness,
            feedback=response.correctness_feedback,
        ),
        "format": Score(
            response.format,
            feedback=response.format_feedback,
        ),
    }


rubric_grader = AsyncRubricGrader(
    rubric,
    judge,
    input_space=spaces.text.text(min_length=1),
)

reward_grader = RewardGrader(
    {
        "exact_match": lambda answer: float(answer == reference_answer),
        "has_citation": lambda answer: float("[1]" in answer),
    },
    input_space=spaces.text.text(min_length=1),
    weights={"exact_match": 1.0, "has_citation": 0.1},
)

grader = AsyncCompositeGrader(
    {"quality": rubric_grader, "verification": reward_grader},
    input_space=spaces.text.text(min_length=1),
    weights={"quality": 0.8, "verification": 0.2},
)

score = await grader.grade(answer)
reward = score.value
```

Rubrics and criteria are strict Pydantic models designed to round-trip through
JSON. A rubric is bound to its `RubricGrader`; the judge receives both the input
and rubric and must return exactly one score per criterion ID. Model selection,
provider SDKs, prompts, sampling settings, retries, caching, and tracing remain
application-owned.

Reward functions receive only the grader input and return a scalar or `Score`.
Their named outputs are combined with a weighted sum by default. Composite
graders use a weighted mean by default and preserve each child grader's full
score tree. Custom aggregation functions are supported by all three families.

`Score` is recursive: components may carry their own feedback, metadata, and
subcomponents. Environments use its scalar value as reward while retaining the
complete grading record under `info["score"]`.

The documentation covers the complete [grader
contract](docs/concepts/graders.md), [rubric
schema](docs/graders/rubrics.md), [programmatic reward
graders](docs/graders/reward-graders.md), [composition and environment
integration](docs/graders/composite-graders.md), [scores and
aggregation](docs/graders/scores-and-aggregation.md), and [representative
examples](docs/graders/examples.md).

## Evaluation

Evaluation benchmarks own a named item collection and an environment factory.
The callback path is intentionally model-backend-neutral and synchronous:

```python
from rolloutlib.evals import Evaluation, run_benchmark
from rolloutlib.evals.benchmarks import gsm8k
from rolloutlib.graders import Score

benchmark = gsm8k([{"question": "What is 6 times 7?", "answer": "#### 42"}])


def evaluate(environment):
    observation, _ = environment.reset()
    response = sample_answer(observation)  # user-owned model code
    _, reward, _, truncated, info = environment.step(response)
    return Evaluation(
        score=Score.from_info(info, default=Score(reward)),
        truncated=truncated,
    )


result = run_benchmark(benchmark, evaluate)
print(result.score)
```

Benchmark runners never receive a grader or rubric. They execute fresh
environments and aggregate the scores those environments produced, ensuring
training and evaluation use the same task semantics.

Built-in AIME and GSM8K environments keep benchmark-specific answer extraction
under `rolloutlib.evals.benchmarks`. Install `rolloutlib[benchmarks]` to load
their conventional Hugging Face datasets.

## Scope

Rolloutlib currently defines environment, space, rollout, grading, and
evaluation contracts. Training operations and backend-specific adapters are
future seams.

## Development

See the [documentation source](docs/index.md), especially
`docs/getting-started.md`, for development, optional integration, and release
setup.

```bash
uv sync
uv run pytest
uv run ruff check rolloutlib tests
uv run pyright
uv build
uv run --group docs mkdocs build --strict
```

## Release

Releases are published to PyPI by GitHub Actions when a tag matching the
package version is pushed:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The repository uses PyPI Trusted Publishing, so the `pypi` GitHub environment
must be registered as a trusted publisher for the repository's release workflow.
