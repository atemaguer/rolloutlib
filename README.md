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
    rubric=item.rubric,
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

Rubrics describe what should be evaluated; graders implement how to evaluate
it. Both are independent from environment action and observation spaces, so a
grader can score an action, trajectory, group, tool result, or arbitrary
application context.

```python
from rolloutlib.graders import CompositeGrader, Criterion, Rubric, Score

rubric = Rubric(
    criteria=(
        Criterion(
            id="correctness",
            description="The answer is correct.",
            weight=1.0,
        ),
        Criterion(
            id="format",
            description="The requested format is followed.",
            weight=0.2,
        ),
    ),
    instructions="Grade only the submitted answer.",
)

grader = CompositeGrader(
    llm_criterion_grader,
    overrides={"correctness": exact_answer_grader},
)

score = grader.score(context, rubric)
score = await grader.ascore(context, rubric)
reward = score.value
```

Each criterion grader receives `(input, criterion)` and may return a number or
a `Score`. `CompositeGrader` executes independent criteria concurrently in its
async path and combines them with a weighted mean by default. `weighted_sum`,
`all_pass`, `asymmetric_mean`, and custom aggregation functions are supported.

`Score` is recursive: its named components are themselves scores and may carry
feedback and metadata. Environments use the scalar value as reward while
retaining the complete grading record:

```python
score = Score(
    0.75,
    {
        "correctness": Score(1.0, feedback="Correct."),
        "format": Score(0.5, feedback="One required heading is missing."),
    },
)
info.update(score.as_info())
assert Score.from_info(info) == score
```

`LLMGrader(sample=..., render=..., parse=...)` supplies a backend-neutral model
boundary. The sampling callable may wrap Tinker, a hosted model API, or local
inference and may be synchronous or asynchronous.

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

```bash
uv sync
uv run pytest
uv run ruff check rolloutlib tests
uv run pyright
uv build
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
