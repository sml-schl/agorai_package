# API Reference

Complete API reference for all AgorAI modules.

## Table of Contents

- [Core Module: `agorai.aggregate`](#agoraiaggregate)
- [Bias Module: `agorai.bias`](#agoraibias)
- [Council Module: `agorai.council`](#agoraicouncil)
- [Testing Module: `agorai.testing`](#agoraitesting)
- [Queue Module: `agorai.queue`](#agoraiqueue)
- [Synthesis Module: `agorai.synthesis`](#agoraisynthesis)
- [Utilities Module: `agorai.utils`](#agoraiutils)

---

## `agorai.aggregate`

Core aggregation functionality.

### `aggregate()`

Main aggregation function that combines agent utilities using a specified method.

**Signature:**
```python
aggregate(
    utilities: Union[List[List[float]], np.ndarray],
    method: str = "majority",
    **method_params
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `utilities` | `List[List[float]]` or `np.ndarray` | *required* | 2D array of utilities (n_agents × n_candidates) |
| `method` | `str` | `"majority"` | Aggregation method name or alias |
| `**method_params` | `Any` | - | Method-specific parameters (e.g., `epsilon` for Atkinson) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `winner` | `int` | Index of winning candidate |
| `scores` | `List[float]` | Aggregated scores for each candidate |
| `method` | `str` | Method used (resolved from alias if applicable) |
| `utilities` | `List[List[float]]` | Original utilities (echoed back) |

**Example:**
```python
from agorai.aggregate import aggregate

utilities = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
result = aggregate(utilities, method="fair", epsilon=1.0)
print(f"Winner: {result['winner']}")
```

---

### `list_methods()`

List all available aggregation methods.

**Signature:**
```python
list_methods() -> List[str]
```

**Returns:**
- `List[str]`: List of method names

**Example:**
```python
from agorai.aggregate import list_methods

methods = list_methods()
print(f"Available methods: {', '.join(methods)}")
```

---

### `register_method()`

Register a custom aggregation method.

**Signature:**
```python
register_method(
    name: str,
    function: Callable,
    description: Optional[str] = None
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Name for the method |
| `function` | `Callable` | *required* | Function that implements the method |
| `description` | `str` | `None` | Optional description |

**Function Signature:**
The custom function must have signature:
```python
def custom_method(utilities: List[List[float]], **params) -> Dict[str, Any]:
    # Must return dict with 'winner' and 'scores' keys
    return {'winner': int, 'scores': List[float], 'method': str}
```

**Example:**
```python
from agorai.aggregate import register_method, aggregate
import numpy as np

def harmonic_mean(utilities, **params):
    utils = np.array(utilities)
    scores = []
    for col in utils.T:
        scores.append(len(col) / np.sum(1.0 / np.maximum(col, 0.1)))
    return {'winner': int(np.argmax(scores)), 'scores': scores, 'method': 'harmonic'}

register_method('harmonic', harmonic_mean, description='Harmonic mean aggregation')
result = aggregate([[0.8, 0.2], [0.6, 0.4]], method='harmonic')
```

---

### `resolve_alias()`

Resolve an alias to its underlying method name.

**Signature:**
```python
resolve_alias(alias: str) -> str
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `alias` | `str` | Alias to resolve |

**Returns:**
- `str`: Resolved method name

**Example:**
```python
from agorai.aggregate import resolve_alias

method = resolve_alias('fair')
print(f"'fair' resolves to: {method}")  # atkinson
```

---

## `agorai.bias`

Bias detection and mitigation through multi-perspective analysis.

### `mitigate_bias()`

Analyze content from multiple perspectives to detect and mitigate bias.

**Signature:**
```python
mitigate_bias(
    input_text: str,
    input_image: Optional[str] = None,
    aggregation_method: str = "fair",
    num_perspectives: int = 5,
    provider: str = "openai",
    model: str = "gpt-4",
    cultural_regions: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_text` | `str` | *required* | Text content to analyze |
| `input_image` | `str` | `None` | Optional image URL or base64 |
| `aggregation_method` | `str` | `"fair"` | Aggregation method (see [aliases](../core/aliases.md)) |
| `num_perspectives` | `int` | `5` | Number of perspectives. Range: [3, 10] |
| `provider` | `str` | `"openai"` | LLM provider: `"openai"`, `"anthropic"`, `"ollama"` |
| `model` | `str` | `"gpt-4"` | Model name |
| `cultural_regions` | `List[str]` | `None` | Specific regions; if `None`, auto-selects |
| `temperature` | `float` | `0.7` | LLM temperature. Range: [0.0, 2.0] |
| `return_individual` | `bool` | `False` | Return individual perspective results |
| `fairness_metrics` | `List[str]` | `["demographic_parity", "equalized_odds"]` | Metrics to compute |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | `"biased"`, `"not_biased"`, or `"uncertain"` |
| `confidence` | `float` | Confidence score [0.0, 1.0] |
| `fairness_metrics` | `Dict` | Computed fairness metrics |
| `perspectives_used` | `List[str]` | Cultural regions used |
| `aggregation_metadata` | `Dict` | Aggregation details |
| `individual_results` | `List[Dict]` | Per-perspective results (if `return_individual=True`) |

**Example:**
```python
from agorai.bias import mitigate_bias

result = mitigate_bias(
    input_text="Is this hiring decision fair?",
    aggregation_method="minority-focused",
    num_perspectives=7,
    return_individual=True
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")

for perspective in result['individual_results']:
    print(f"{perspective['region']}: {perspective['assessment']}")
```

---

## `agorai.council`

Automatic council creation for multi-perspective decision-making.

### `create_council()`

Automatically generate a diverse council of perspectives.

**Signature:**
```python
create_council(
    context: str,
    num_perspectives: int = 5,
    diversity_dimension: str = "stakeholder",
    provider: str = "openai",
    model: str = "gpt-4",
    region_constraint: Optional[List[str]] = None,
    **kwargs
) -> Council
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str` | *required* | Decision context description |
| `num_perspectives` | `int` | `5` | Number of perspectives. Range: [3, 15] |
| `diversity_dimension` | `str` | `"stakeholder"` | Options: `"stakeholder"`, `"cultural"`, `"expertise"`, `"demographic"`, `"ideological"` |
| `provider` | `str` | `"openai"` | LLM provider |
| `model` | `str` | `"gpt-4"` | Model name |
| `region_constraint` | `List[str]` | `None` | For cultural diversity, specific regions |
| `temperature` | `float` | `0.9` | Temperature. Range: [0.0, 2.0] |
| `ensure_minority` | `bool` | `True` | Ensure minority perspectives |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `perspectives` | `List[Perspective]` | Generated perspectives |
| `diversity_score` | `float` | Diversity measure [0.0, 1.0] |
| `coverage_metrics` | `Dict` | Coverage statistics |

**Example:**
```python
from agorai.council import create_council

council = create_council(
    context="content moderation for global platform",
    num_perspectives=7,
    diversity_dimension="cultural",
    ensure_minority=True
)

print(f"Diversity score: {council.diversity_score:.2f}")
print(f"Perspectives: {[p.name for p in council.perspectives]}")
```

---

### `synthesize_with_council()`

Use a council to make a decision.

**Signature:**
```python
synthesize_with_council(
    prompt: str,
    council: Council,
    aggregation_method: str = "fair",
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Decision prompt |
| `council` | `Council` | *required* | Council from `create_council()` |
| `aggregation_method` | `str` | `"fair"` | Aggregation method |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | Aggregated decision |
| `confidence` | `float` | Confidence [0.0, 1.0] |
| `perspectives_used` | `List[str]` | Perspective names |
| `individual_results` | `List[Dict]` | Per-perspective results |

**Example:**
```python
from agorai.council import create_council, synthesize_with_council

council = create_council(context="hiring decision", num_perspectives=5)

result = synthesize_with_council(
    prompt="Should we hire candidate X?",
    council=council,
    aggregation_method="fair"
)

print(f"Decision: {result['decision']}")
```

---

## `agorai.testing`

Counterfactual testing for bias and robustness evaluation.

### `test_counterfactuals()`

Test model robustness to protected attribute changes.

**Signature:**
```python
test_counterfactuals(
    input_text: str,
    input_image: Optional[str] = None,
    protected_attributes: Optional[List[str]] = None,
    aggregation_method: str = "fair",
    num_perspectives: int = 5,
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_text` | `str` | *required* | Original text |
| `input_image` | `str` | `None` | Optional image |
| `protected_attributes` | `List[str]` | `None` | Attributes to test; if `None`, tests all: `["ethnicity", "gender", "age", "religion", "nationality"]` |
| `aggregation_method` | `str` | `"fair"` | Aggregation method |
| `num_perspectives` | `int` | `5` | Perspectives per test |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `original_result` | `Dict` | Result for original input |
| `counterfactual_results` | `List[Dict]` | Results for each counterfactual |
| `spurious_correlations` | `List[Dict]` | Detected spurious correlations |
| `robustness_score` | `float` | Overall robustness [0.0, 1.0] |
| `edge_cases` | `List[Dict]` | Identified edge cases |

**Example:**
```python
from agorai.testing import test_counterfactuals

result = test_counterfactuals(
    input_text="Applicant with excellent qualifications",
    protected_attributes=["ethnicity", "gender"],
    aggregation_method="fair"
)

print(f"Robustness score: {result['robustness_score']:.2f}")

if result['spurious_correlations']:
    print("Spurious correlations detected:")
    for corr in result['spurious_correlations']:
        print(f"  - {corr['attribute']}: {corr['description']}")
```

---

## `agorai.queue`

Batch processing for production workflows.

### `process_queue()`

Process multiple aggregation requests from a file.

**Signature:**
```python
process_queue(
    requests_file: str,
    method: str = "majority",
    metrics: Optional[List[str]] = None,
    **method_params
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_file` | `str` | *required* | Path to JSON file |
| `method` | `str` | `"majority"` | Aggregation method |
| `metrics` | `List[str]` | `None` | Metrics to compute: `["fairness", "efficiency", "agreement"]` |
| `**method_params` | `Any` | - | Method-specific parameters |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[Dict]` | Per-item results |
| `summary` | `Dict` | Aggregate statistics |
| `num_requests` | `int` | Number of requests processed |
| `source_name` | `str` | Name from file |

**Example:**
```python
from agorai.queue import process_queue

results = process_queue(
    requests_file="daily_decisions.json",
    method="atkinson",
    metrics=["fairness", "efficiency"],
    epsilon=1.0
)

print(f"Processed: {results['num_requests']} requests")
print(f"Gini: {results['summary']['fairness']['gini_coefficient']:.3f}")
```

---

### `compare_methods_on_queue()`

Compare multiple aggregation methods on the same queue.

**Signature:**
```python
compare_methods_on_queue(
    requests_file: str,
    methods: List[Union[str, Dict]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_file` | `str` | *required* | Path to JSON file |
| `methods` | `List[Union[str, Dict]]` | *required* | Method names or dicts with `{"name": str, "params": dict}` |
| `metrics` | `List[str]` | `None` | Metrics to compute |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `methods` | `List[Dict]` | Results for each method |
| `rankings` | `Dict[str, List[str]]` | Rankings by metric |
| `num_requests` | `int` | Number of requests |

**Example:**
```python
from agorai.queue import compare_methods_on_queue

comparison = compare_methods_on_queue(
    requests_file="test_set.json",
    methods=["majority", "atkinson", "maximin"],
    metrics=["fairness", "efficiency"]
)

fairest = comparison['rankings']['fairness_gini_coefficient'][0]
print(f"Fairest method: {fairest}")
```

---

## `agorai.synthesis`

LLM-based opinion synthesis.

### `synthesize()`

Synthesize opinions from multiple LLM agents.

**Signature:**
```python
synthesize(
    prompt: str,
    agents: List[Agent],
    aggregation_method: str = "majority",
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Question or decision prompt |
| `agents` | `List[Agent]` | *required* | List of Agent objects |
| `aggregation_method` | `str` | `"majority"` | Aggregation method |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | Synthesized decision |
| `confidence` | `float` | Confidence [0.0, 1.0] |
| `agent_responses` | `List[Dict]` | Individual agent responses |

**Example:**
```python
from agorai.synthesis import Agent, synthesize

agents = [
    Agent(provider="openai", model="gpt-4"),
    Agent(provider="anthropic", model="claude-3-5-sonnet-20241022"),
    Agent(provider="ollama", model="llama3.2")
]

result = synthesize(
    prompt="Should we implement feature X?",
    agents=agents,
    aggregation_method="fair"
)

print(f"Decision: {result['decision']}")
```

---

### `Agent`

Agent configuration for synthesis.

**Signature:**
```python
Agent(
    provider: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | *required* | `"openai"`, `"anthropic"`, `"ollama"` |
| `model` | `str` | *required* | Model name |
| `system_prompt` | `str` | `None` | Custom system prompt |
| `temperature` | `float` | `0.7` | Temperature. Range: [0.0, 2.0] |

**Example:**
```python
from agorai.synthesis import Agent

agent = Agent(
    provider="openai",
    model="gpt-4",
    system_prompt="You are a cautious decision-maker who prioritizes safety.",
    temperature=0.5
)
```

---

## `agorai.utils`

Utility functions.

### `calculate_fairness_metrics()`

Calculate fairness metrics for utilities and scores.

**Signature:**
```python
calculate_fairness_metrics(
    utilities: List[List[float]],
    scores: List[float]
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `utilities` | `List[List[float]]` | Agent utilities |
| `scores` | `List[float]` | Aggregated scores |

**Returns:**

| Metric | Description | Range |
|--------|-------------|-------|
| `gini_coefficient` | Gini coefficient (0 = equal, 1 = unequal) | [0.0, 1.0] |
| `atkinson_index` | Atkinson index (0 = equal, 1 = unequal) | [0.0, 1.0] |
| `variance` | Variance of utilities | [0.0, ∞) |
| `coefficient_of_variation` | Normalized variance | [0.0, ∞) |

**Example:**
```python
from agorai.utils import calculate_fairness_metrics

utilities = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
scores = [0.57, 0.43]

metrics = calculate_fairness_metrics(utilities, scores)
print(f"Gini: {metrics['gini_coefficient']:.3f}")
```

---

### `calculate_efficiency_metrics()`

Calculate efficiency metrics.

**Signature:**
```python
calculate_efficiency_metrics(
    utilities: List[List[float]],
    winner: int
) -> Dict[str, float]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `utilities` | `List[List[float]]` | Agent utilities |
| `winner` | `int` | Winning candidate index |

**Returns:**

| Metric | Description |
|--------|-------------|
| `social_welfare` | Sum of utilities for winner |
| `utilitarian_welfare` | Mean utility for winner |
| `pareto_efficiency` | 1.0 if Pareto optimal, 0.0 otherwise |

**Example:**
```python
from agorai.utils import calculate_efficiency_metrics

utilities = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
winner = 0

metrics = calculate_efficiency_metrics(utilities, winner)
print(f"Social welfare: {metrics['social_welfare']:.2f}")
```

---

## Type Definitions

### `Utilities`

```python
Utilities = Union[List[List[float]], np.ndarray]
```

2D array where:
- Rows represent agents
- Columns represent candidates
- Values are utilities in range [0.0, 1.0]

---

### `AggregationResult`

```python
class AggregationResult(TypedDict):
    winner: int
    scores: List[float]
    method: str
    utilities: List[List[float]]
```

---

### `Council`

```python
class Council:
    perspectives: List[Perspective]
    diversity_score: float
    coverage_metrics: Dict[str, Any]
```

---

### `Perspective`

```python
class Perspective:
    name: str
    system_prompt: str
    provider: str
    model: str
    values: List[str]
```

---

## Error Handling

### Common Exceptions

**`ValueError`**
- Invalid parameters (e.g., negative epsilon)
- Malformed utilities (inconsistent shapes)
- Unknown method names

```python
from agorai.aggregate import aggregate

try:
    result = aggregate([[0.8, 0.2], [0.3]], method="majority")
except ValueError as e:
    print(f"Error: {e}")  # Inconsistent number of candidates
```

**`APIError`**
- LLM API failures
- Authentication errors
- Rate limiting

```python
from agorai.bias import mitigate_bias
from agorai.exceptions import APIError

try:
    result = mitigate_bias("Is this biased?")
except APIError as e:
    print(f"API Error: {e}")
```

**`FileNotFoundError`**
- Missing request files

```python
from agorai.queue import process_queue

try:
    result = process_queue("nonexistent.json", method="majority")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

---

## Environment Variables

See [Configuration](configuration.md) for environment variable documentation.

---

## See Also

- [Aggregation Methods](../core/aggregation.md) - Detailed method documentation
- [Mechanism Aliases](../core/aliases.md) - Intuitive method names
- [Configuration](configuration.md) - Settings and environment variables
- [Examples](../../examples/) - Code examples
