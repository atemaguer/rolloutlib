# Spaces

Spaces describe which values are valid. Environments expose an
`observation_space` and an `action_space`, while graders expose an
`input_space`. The same space type can serve any of these roles.

Rolloutlib accepts all Gymnasium spaces and adds common language-model values:

- token identifiers and token sequences;
- Unicode text;
- role-tagged messages and chats;
- structured tool calls.

```python
import gymnasium as gym

from rolloutlib import spaces

text = spaces.text.text(min_length=1, max_length=1_000)
chat = spaces.messages.chat(min_length=1)
tool_call = spaces.tools.call(
    {
        "search": gym.spaces.Dict(
            {"query": spaces.text.text(min_length=1)}
        )
    }
)
```

Structured spaces use ordinary dictionaries and lists. Pydantic performs
strict validation at the boundary; applications do not need framework-specific
message or tool-call objects.

Every domain space supports membership checks, seeded sampling, and Gymnasium
JSON conversion. A text space's sampling alphabet controls generated samples,
not the Unicode strings accepted by the space.

A grader input space describes the complete grading input. For example, a
grader may accept only a response string, or it may accept a structured value
containing the task, response, reference answer, and rollout trace. Rolloutlib
does not prescribe one universal grading record; it uses spaces to make each
grader's chosen contract explicit and enforceable.
