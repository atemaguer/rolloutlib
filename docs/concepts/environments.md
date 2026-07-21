# Environments

An environment defines the task that a policy interacts with. Rolloutlib uses
Gymnasium's environment contract so existing Gymnasium tools and expectations
continue to apply.

## Synchronous environments

A synchronous environment is a `gymnasium.Env`. Its two central operations are:

- `reset()` starts an episode and returns the first observation plus an
  information dictionary.
- `step(action)` advances the task and returns the next observation, scalar
  reward, `terminated`, `truncated`, and information.

`terminated` means the task reached an outcome. `truncated` means an external
limit ended the episode. Keeping those meanings distinct matters to training
algorithms.

## Wrapping Gymnasium environments for language agents

`ChatObservationWrapper` and `ToolCallActionWrapper` use Gymnasium's standard
`ObservationWrapper` and `ActionWrapper` APIs to present an existing
environment to a language agent. `wrap_language_env` composes both wrappers for
the common case and handles JSON serialization, multimodal message
construction, and action validation.

```python
import gymnasium as gym

from rolloutlib import wrappers


env = wrappers.wrap_language_env(
    gym.make("ExistingEnv-v0", render_mode="rgb_array"),
    include_render=True,
    instructions="Choose the next environment action.",
)
```

With no selector functions, the native observation is encoded as JSON using its
Gymnasium space. For environment-specific state or media, provide only the
selectors:

```python
env = wrappers.wrap_language_env(
    env,
    state=lambda observation: {
        "status": observation["status"],
        "choices": observation["choices"],
    },
    image=lambda observation: observation["pixels"],
    image_alt="Current environment view",
    audio=lambda observation: observation["samples"],
    audio_sample_rate=16_000,
    tool_name="choose",
    argument_name="action",
    tool_description="Choose the next environment action.",
    available_actions=lambda: env.unwrapped.available_actions,
)
```

Image arrays are encoded with Pillow, available through the `media` and
`openai` extras. Audio sample arrays are encoded as WAV. URL and byte inputs are
also accepted.

The outer environment remains a normal Gymnasium environment. Its
`observation_space` is the chat space, its `action_space` is the tool-call
space, and `env.unwrapped` remains the original environment. It can be passed
directly to `rollout`. By default, the action space accepts calls of the form
`{"name": "step", "arguments": {"action": native_action}}` and validates the
argument against the original environment's action space.

### Model-provider policies

The optional OpenAI policy translates chat content, derives strict function
tools from the wrapped action space, parses one function call, and validates it
before the environment receives it:

```python
from openai import OpenAI

from rolloutlib import rollout
from rolloutlib.policies import OpenAIResponsesPolicy

policy = OpenAIResponsesPolicy.from_env(
    env,
    client=OpenAI(),
    model="your-model",
    instructions="Choose exactly one action.",
)
trajectory = rollout(env, policy)
```

Install it with `pip install "rolloutlib[openai]"`. The provider implementation
is optional; environments continue to expose ordinary backend-neutral chats
and tool calls.

### Episode history

`history` adds a Gymnasium wrapper whose observation is the episode's bounded
chat history:

```python
env = wrappers.wrap_language_env(
    env,
    history=32,
    retain_media="latest",
)
```

Actions and resulting observations are appended on each step. By default only
the latest message retains image or audio payloads, which avoids repeatedly
embedding old media in every subsequent observation.

Use the wrapper classes separately to customize their order or insert other
Gymnasium wrappers:

```python
from rolloutlib import wrappers

inner = gym.make("ExistingEnv-v0")
env = wrappers.ChatObservationWrapper(
    inner,
    to_chat,
    observation_space=chat_space,
)
env = wrappers.ToolCallActionWrapper(
    env,
    tool_name="choose",
    argument_name="value",
)
```

Modality selection remains environment-specific because array shape alone
cannot reliably distinguish an image, audio, or ordinary numeric state.
Rolloutlib handles encoding after the application identifies the relevant
fields.

## Composition validation

Rolloutlib uses Gymnasium spaces as contracts between producers and consumers.
`spaces.check_space_compatibility(produced, accepted)` verifies that every value
the producer may emit can be accepted by the consumer. Common Gymnasium scalar
and array spaces, nested `Dict`, `Tuple`, and `Sequence` spaces, and rolloutlib's
language spaces are compared structurally. Unknown custom spaces must compare
equal, so the check remains conservative.

Rollouts validate declared policy spaces against environment spaces before
reset, then validate the actual observations, actions, rewards, termination
flags, and info mappings as the episode runs. Wrappers similarly validate their
transformed values. Use `spaces.check_space_value(space, value, name=...)` when
building another composition point.

## Awaitable operations

`Env` is the single environment contract. Its `reset`, `step`, and `close`
methods may return their normal Gymnasium values directly or awaitables that
resolve to those values. This supports tool execution, remote services,
browsers, and other asynchronous resources without a second hierarchy.

Use `rollout` with immediate operations. Use `arollout` or `arollout_group`
when an environment or policy may perform asynchronous work; those collectors
accept the same `Env` instance and resolve awaitables as needed.

## Single-turn tasks

`SingleTurnEnv` covers tasks where one action completes
the episode. Subclasses define:

- the initial observation;
- how the action is evaluated;
- the terminal observation.

The evaluator may return a scalar or a structured `Score`. Structured scores
become the scalar reward while remaining available under `info["score"]`.

## Adding grading to an environment

Grading that determines reward belongs inside `step`. If an existing
environment does not grade its own terminal state, `GradingWrapper` can
compose it with a grader.

```python
from rolloutlib import wrappers

environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=lambda env, action: (item, env.state, action),
    input_space=grader.input_space,
)
```

By default, grading occurs on terminated or truncated steps, replaces the
environment reward with `Score.value`, and retains the full score in `info`.
When `make_input` is present, `input_space` declares its output contract and is
checked against the grader at construction time. Omit both to grade environment
actions directly. Custom predicates and reward combiners can change those
policies. When using a rubric grader, its rubric is already bound to the grader
rather than passed through the environment wrapper.

See [Composite graders](../graders/composite-graders.md#grading-inside-environments)
for synchronous and asynchronous wrapper examples, terminal-step policies, and
reward combination.
