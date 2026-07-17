# Datasets and evaluation

Datasets provide the work to be performed; they do not define how a policy is
sampled or trained. An `RLDataset` pairs each item with a factory that creates a
fresh environment for that item:

```python
from rolloutlib.datasets import RLDataset


dataset = RLDataset(
    items=training_items,
    make_env=make_training_environment,
)
```

An item is pre-rollout context, such as a prompt, task specification, or
reference material. It is not a completed trajectory. Batch size, shuffling,
repetition, and the number of rollouts per item belong to the training system.

## Benchmarks

A `Benchmark` gives a named evaluation collection an environment factory. It
can share the same task implementation as training while using held-out items:

```python
from rolloutlib.evals import Benchmark


benchmark = Benchmark(
    name="held-out-math",
    items=evaluation_items,
    make_env=make_evaluation_environment,
)
```

Evaluation is intentionally model-backend-neutral. The application supplies a
callback that runs its policy against one fresh environment and returns an
`Evaluation` containing a `Score`:

```python
from rolloutlib.evals import Evaluation, run_benchmark


def evaluate(environment) -> Evaluation:
    trajectory = run_my_policy(environment)
    return Evaluation(score=score_trajectory(trajectory))


result = run_benchmark(benchmark, evaluate)
```

`BenchmarkResult` reports the aggregate score, criterion component averages,
completion and truncation counts, errors, elapsed time, and per-item records.
This keeps evaluation results inspectable without imposing a model SDK,
optimizer, logging system, or experiment tracker.

Scores may come directly from an environment, from a grader applied to the
trajectory, or from both. The important boundary is that the benchmark owns the
items and task factory, while the application owns policy execution.
