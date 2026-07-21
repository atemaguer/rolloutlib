# rolloutlib

Rolloutlib is a collection of interoperable Python libraries for agentic
reinforcement-learning (RL) post-training.

It defines standard APIs for the interactive part of agentic RL: environments
define tasks and state transitions; multimodal spaces describe observations and
actions; policies sample language-model actions; rollout runners record
trajectories; and graders produce reward and evaluation signals.

These contracts let the same task, trajectory, and reward semantics work across
model providers, rollout workers, RL trainers, and evaluation harnesses.
Rolloutlib is Gymnasium-native, so existing environments remain usable while
wrappers expose their observations and actions to language agents.

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

The same `Env` supports asynchronous implementations:

```python
from rolloutlib import Env


class ToolEnv(Env):
    action_space = ...
    observation_space = ...

    async def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return observation, {}

    async def step(self, action):
        result = await execute_tool(action)
        return next_observation, reward, terminated, truncated, {"result": result}
```

`rollout` requires immediate environment and policy results. `arollout` and
`arollout_group` accept the same objects and resolve awaitable operations, so
no parallel environment hierarchy or bridge class is needed.

Grading that determines reward belongs inside `step`. Single-turn environments
may return a `Score` directly from `evaluate`; existing environments can be
composed with a grading wrapper:

```python
from rolloutlib import wrappers

environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=lambda env, action: (item, env.state, action),
    input_space=grader.input_space,
)
```

`GradingWrapper` follows the same contract: its step result is immediate when
both collaborators are immediate and awaitable otherwise.
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

Existing Gymnasium environments can be presented to language agents with
rolloutlib's Gymnasium-native wrappers:

```python
import gymnasium as gym

from rolloutlib import wrappers

env = wrappers.wrap_language_env(
    gym.make("CartPole-v1", render_mode="rgb_array"),
    include_render=True,
    instructions="Actions: 0 pushes left; 1 pushes right.",
)
```

Without configuration, native observations are JSON-serialized from their
Gymnasium space and native actions are exposed as one validated tool call.
State, image, and audio selectors cover environments that need a more semantic
or multimodal presentation without requiring callers to build message
dictionaries or data URLs.

The transformed environment remains a normal `gymnasium.Env`. Provider policies
can derive tool definitions directly from it:

```python
from openai import OpenAI

from rolloutlib import rollout
from rolloutlib.policies import OpenAIResponsesPolicy

policy = OpenAIResponsesPolicy.from_env(
    env,
    client=OpenAI(),
    model="your-model",
)
trajectory = rollout(env, policy)
```

See the
[environment guide](docs/concepts/environments.md#wrapping-gymnasium-environments-for-language-agents)
for validation and a multimodal example.

The opt-in OpenAI chess integration exercises this path against the real
`BulletChess-v0` Gymnasium environment. It runs 20 self-play steps with
`gpt-5.6-luna` at reasoning effort `none`, sends the rendered board and legal
moves to the OpenAI Responses API, applies every returned tool call, and
verifies that the environment accepted every move:

```console
uv sync --extra openai-chess
export OPENAI_API_KEY=...
RUN_OPENAI_CHESS_INTEGRATION=1 uv run pytest \
  tests/test_openai_chess_integration.py -q
```

Set `OPENAI_CHESS_MODEL` to override the default model. This test makes 20 paid
API requests and is skipped during ordinary test runs.

The same wrapper and OpenAI policy path has also been validated against
FinRL's continuous stock actions, FinRL-Meta's market-impact portfolio
environment, and the FinRL portfolio environment consumed by FinRL-trading.
See the [FinRL integration guide](docs/integrations/finrl.md) for the minimal
wrapper code and reproducible synthetic-market tests.

## Rollouts

The rollout layer records environment interactions while leaving sampling to
user code. `Policy` is one callable contract whose result may be immediate or
awaitable. It may return a raw action or a
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

For RL groups, the same collectors also accept `batch_policy`. The synchronous
collector begins through Gymnasium `SyncVectorEnv` and calls the batch policy
once per active wave. If members finish at different times, it transparently
continues only the unfinished environments:

```python
from rolloutlib import rollout_group

group = rollout_group(
    item,
    make_env,
    batch_policy=lambda observations: model.generate_batch(observations),
    num_rollouts=8,
)
```

The async collector follows the same interface. It concurrently steps active
environments, removes completed slots, and continues with the remainder:

```python
from rolloutlib import arollout_group

group = await arollout_group(
    item,
    make_async_env,
    batch_policy=async_batch_policy,
    num_rollouts=8,
)
```

Each group function accepts exactly one of the existing scalar `policy`
argument and `batch_policy`. Both modes retain one `Trajectory` per environment
and validate batch cardinality, declared spaces, returned actions,
observations, rewards, and termination flags. The reproducible collector
comparison lives in `benchmarks/rollout_group_throughput.py`.

Backend-specific policies are ordinary user code. For example, a Tinker
policy can build a generation prompt with a Tinker renderer, call
`SamplingClient.sample` (or `sample_async`), parse the returned tokens, and
return `PolicyOutput`. This keeps Tinker optional while preserving the same
rollout contract.

## Graders

`Grader` defines one operation: `grade(input) -> Score | Awaitable[Score]`.
Every grader has an `input_space`
describing its complete input, so it can safely score a response, action,
trajectory, tool trace, or richer application-defined record.

Rolloutlib provides three grader families:

- `RubricGrader` applies a portable, human-defined rubric through a user-owned
  judge, commonly an LLM judge.
- `RewardGrader` evaluates named programmatic reward functions.
- `CompositeGrader` combines complete graders while preserving their nested
  results.

Each family accepts synchronous or asynchronous user-owned operations.

```python
from rolloutlib import spaces
from rolloutlib.graders import (
    CompositeGrader,
    RubricGrader,
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


rubric_grader = RubricGrader(
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

grader = CompositeGrader(
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

Rolloutlib focuses on task definition and experience generation for agentic RL:
sampling actions, recording trajectories, producing reward signals, and
evaluating policies with the same environment semantics used during training.
It does not implement optimization algorithms or a distributed training
runtime. Model providers and RL trainers integrate at the policy and trajectory
boundaries.

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
