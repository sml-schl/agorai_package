# Extending AgorAI: Developer Guide

This guide covers how to extend AgorAI with custom aggregation methods, modify configurations, and integrate with your own systems.

## Table of Contents

1. [Adding New Aggregation Methods](#adding-new-aggregation-methods)
2. [Configuring LLM Providers](#configuring-llm-providers)
3. [Creating Custom Metrics](#creating-custom-metrics)
4. [Extending the Queue System](#extending-the-queue-system)
5. [Custom Visualization](#custom-visualization)
6. [Integration Patterns](#integration-patterns)

---

## Adding New Aggregation Methods

### Quick Start

The simplest way to add a new aggregation method:

```python
from agorai.aggregate import register_method
import numpy as np

def my_custom_method(utilities, **params):
    """My custom aggregation method.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    **params
        Method-specific parameters

    Returns
    -------
    Dict[str, Any]
        Result with 'winner' (int) and 'scores' (List[float])
    """
    # Convert to numpy for easier computation
    utils = np.array(utilities)

    # Example: Weighted geometric mean
    alpha = params.get('alpha', 0.5)

    # Compute scores (geometric mean with weights)
    scores = []
    for candidate_idx in range(utils.shape[1]):
        candidate_utils = utils[:, candidate_idx]
        # Add small epsilon to avoid log(0)
        safe_utils = np.maximum(candidate_utils, 1e-10)
        geo_mean = np.exp(np.mean(np.log(safe_utils)) * alpha)
        scores.append(float(geo_mean))

    winner = int(np.argmax(scores))

    return {
        'winner': winner,
        'scores': scores,
        'method': 'my_custom_method',
        'params': params
    }

# Register the method
register_method('my_custom', my_custom_method)

# Use it!
from agorai.aggregate import aggregate

utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
result = aggregate(utilities, method='my_custom', alpha=0.7)
print(f"Winner: {result['winner']}")
```

### Method Requirements

Your aggregation method must:

1. **Accept `utilities` as first parameter:** `List[List[float]]` (n_agents × n_candidates)
2. **Accept `**params` for method-specific parameters**
3. **Return a dictionary with:**
   - `'winner'` (int): Index of winning candidate
   - `'scores'` (List[float]): Aggregated scores for all candidates
   - `'method'` (str): Method name
   - Optionally: `'params'`, `'metadata'`, or other info

### Complete Example: Harmonic Mean Aggregation

```python
from agorai.aggregate import register_method
import numpy as np

def harmonic_mean_aggregation(utilities, min_threshold=0.1, **params):
    """Aggregation using harmonic mean (emphasizes low utilities).

    The harmonic mean gives more weight to lower values, making it
    useful for ensuring minimum satisfaction across agents.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix (n_agents × n_candidates)
    min_threshold : float
        Minimum utility value to avoid division by zero (default: 0.1)

    Returns
    -------
    Dict[str, Any]
        Aggregation result with winner and scores

    Properties
    -------
    - Emphasizes fairness (penalizes low utilities)
    - Not Pareto efficient
    - Sensitive to outliers

    Examples
    --------
    >>> utilities = [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]]
    >>> result = harmonic_mean_aggregation(utilities)
    >>> print(result['winner'])
    0
    """
    utils = np.array(utilities)
    n_agents, n_candidates = utils.shape

    scores = []
    for candidate_idx in range(n_candidates):
        candidate_utils = utils[:, candidate_idx]

        # Apply threshold to avoid division by zero
        safe_utils = np.maximum(candidate_utils, min_threshold)

        # Harmonic mean: H = n / sum(1/x_i)
        harmonic_mean = n_agents / np.sum(1.0 / safe_utils)
        scores.append(float(harmonic_mean))

    winner = int(np.argmax(scores))

    return {
        'winner': winner,
        'scores': scores,
        'method': 'harmonic_mean',
        'params': {'min_threshold': min_threshold, **params}
    }

# Register
register_method('harmonic_mean', harmonic_mean_aggregation)

# Test
utilities = [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]]
result = aggregate(utilities, method='harmonic_mean', min_threshold=0.15)
print(f"Winner: Candidate {result['winner']}")
print(f"Scores: {result['scores']}")
```

### Testing Your Method

Always test your custom method thoroughly:

```python
import pytest
from agorai.aggregate import aggregate, register_method

def test_my_custom_method():
    """Test custom aggregation method."""

    # Test 1: Basic functionality
    utilities = [[0.8, 0.2], [0.3, 0.7]]
    result = aggregate(utilities, method='my_custom')

    assert 'winner' in result
    assert 'scores' in result
    assert len(result['scores']) == 2
    assert 0 <= result['winner'] < 2

    # Test 2: Clear winner
    utilities = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    result = aggregate(utilities, method='my_custom')
    assert result['winner'] == 0  # Should choose first candidate

    # Test 3: Parameter handling
    result = aggregate(utilities, method='my_custom', alpha=0.5)
    assert result['params']['alpha'] == 0.5

    # Test 4: Edge cases
    # Single agent
    utilities = [[0.6, 0.4]]
    result = aggregate(utilities, method='my_custom')
    assert result['winner'] == 0

    # Tied utilities
    utilities = [[0.5, 0.5], [0.5, 0.5]]
    result = aggregate(utilities, method='my_custom')
    assert result['winner'] in [0, 1]  # Either is valid

    print("✓ All tests passed!")

# Run tests
test_my_custom_method()
```

### Best Practices

1. **Handle edge cases:**
   ```python
   def robust_method(utilities, **params):
       utils = np.array(utilities)

       # Check for empty input
       if utils.size == 0:
           raise ValueError("Empty utilities matrix")

       # Check for consistent dimensions
       if len(set(len(row) for row in utilities)) > 1:
           raise ValueError("Inconsistent number of candidates")

       # Handle zero/negative utilities if needed
       safe_utils = np.maximum(utils, 1e-10)

       # ... rest of method
   ```

2. **Document properties:**
   ```python
   def my_method(utilities, **params):
       """My aggregation method.

       Properties
       ----------
       - Pareto efficient: Yes/No
       - Strategy-proof: Yes/No
       - Monotonic: Yes/No
       - Anonymous: Yes/No

       Computational Complexity
       ------------------------
       O(n * m) where n=agents, m=candidates
       """
   ```

3. **Provide examples in docstring:**
   ```python
   def my_method(utilities, **params):
       """
       Examples
       --------
       >>> utilities = [[0.8, 0.2], [0.3, 0.7]]
       >>> result = my_method(utilities)
       >>> print(result['winner'])
       0
       """
   ```

4. **Return metadata for debugging:**
   ```python
   return {
       'winner': winner,
       'scores': scores,
       'method': 'my_method',
       'params': params,
       'metadata': {
           'iterations': num_iterations,
           'convergence': convergence_value,
           'warnings': warnings_list
       }
   }
   ```

---

## Configuring LLM Providers

### Changing Default LLM Provider

The synthesis module supports multiple LLM providers. Here's how to configure each:

### OpenAI (GPT Models)

```python
from agorai.synthesis import Agent

# Basic configuration
agent = Agent(
    provider="openai",
    model="gpt-4",
    api_key="sk-...",  # Or set OPENAI_API_KEY environment variable
)

# Advanced configuration
agent = Agent(
    provider="openai",
    model="gpt-4-turbo-preview",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1000,
    system_prompt="You are a fair and balanced decision maker."
)

# Use with synthesis
from agorai.synthesis import synthesize

result = synthesize(
    prompt="Should we implement this feature?",
    agents=[agent],
    candidates=["Yes", "No", "Maybe"],
    aggregation_method="majority"
)
```

### Anthropic (Claude Models)

```python
from agorai.synthesis import Agent

# Claude 3.5 Sonnet (recommended)
agent = Agent(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",  # Or set ANTHROPIC_API_KEY environment variable
)

# Claude with custom configuration
agent = Agent(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
    temperature=0.5,
    max_tokens=2000,
    system_prompt="You are a thoughtful analyst."
)
```

### Ollama (Local Models)

```python
from agorai.synthesis import Agent

# Default (localhost:11434)
agent = Agent(
    provider="ollama",
    model="llama3.2",
)

# Custom host/port
agent = Agent(
    provider="ollama",
    model="llama3.2",
    base_url="http://192.168.1.100:11434"
)

# With parameters
agent = Agent(
    provider="ollama",
    model="mistral",
    temperature=0.8,
    num_predict=500,
    system_prompt="You are an expert evaluator."
)
```

### Google (Gemini Models)

```python
from agorai.synthesis import Agent

agent = Agent(
    provider="google",
    model="gemini-pro",
    api_key="...",  # Or set GOOGLE_API_KEY environment variable
)
```

### Multi-Provider Setup

Create diverse councils with multiple providers:

```python
from agorai.synthesis import Agent, Council

# Create agents from different providers
agents = [
    Agent(provider="openai", model="gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
    Agent(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key=os.getenv("ANTHROPIC_API_KEY")),
    Agent(provider="ollama", model="llama3.2"),
    Agent(provider="ollama", model="mistral"),
]

# Create council
council = Council(
    agents=agents,
    aggregation_method="atkinson",
    aggregation_params={"epsilon": 1.0}
)

# Make decision
result = council.decide(
    prompt="Evaluate this proposal...",
    candidates=["Approve", "Reject", "Revise"]
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Environment Variables

Set API keys via environment variables:

```bash
# In your .env file or shell
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Then use without explicit keys:

```python
from agorai.synthesis import Agent

# Automatically uses environment variables
agent_gpt = Agent(provider="openai", model="gpt-4")
agent_claude = Agent(provider="anthropic", model="claude-3-5-sonnet-20241022")
```

### Custom System Prompts

Customize agent behavior with system prompts:

```python
# Fairness-focused agent
fair_agent = Agent(
    provider="openai",
    model="gpt-4",
    system_prompt="""You are a fairness-focused evaluator.
    Always consider how decisions impact all stakeholders equally.
    Prioritize equity over efficiency when there's a tradeoff."""
)

# Efficiency-focused agent
efficient_agent = Agent(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    system_prompt="""You are an efficiency-focused evaluator.
    Prioritize outcomes that maximize total benefit.
    Consider practical implementation and resource constraints."""
)

# Risk-averse agent
cautious_agent = Agent(
    provider="ollama",
    model="llama3.2",
    system_prompt="""You are a risk-averse evaluator.
    Carefully consider potential downsides and edge cases.
    Prefer conservative, well-tested approaches."""
)

# Use diverse agents in council
council = Council(agents=[fair_agent, efficient_agent, cautious_agent])
```

---

## Creating Custom Metrics

### Adding Metrics to Queue Processing

Extend the metrics system with your own calculations:

```python
from agorai.queue.metrics import (
    calculate_fairness_metrics,
    calculate_efficiency_metrics,
    calculate_agreement_metrics
)
import numpy as np

def calculate_custom_metrics(utilities, winner):
    """Calculate custom metrics for aggregation results.

    Parameters
    ----------
    utilities : List[List[float]]
        Utility matrix
    winner : int
        Winning candidate index

    Returns
    -------
    Dict[str, float]
        Dictionary of metric values
    """
    utils = np.array(utilities)
    winner_utils = utils[:, winner]

    metrics = {}

    # Custom metric 1: Satisfaction ratio
    # Ratio of agents with utility > 0.5
    metrics['satisfaction_ratio'] = float(np.mean(winner_utils > 0.5))

    # Custom metric 2: Worst-case margin
    # Difference between worst agent's utility and next-worst
    sorted_utils = np.sort(winner_utils)
    if len(sorted_utils) >= 2:
        metrics['worst_case_margin'] = float(sorted_utils[1] - sorted_utils[0])
    else:
        metrics['worst_case_margin'] = 0.0

    # Custom metric 3: Utility range
    metrics['utility_range'] = float(np.max(winner_utils) - np.min(winner_utils))

    # Custom metric 4: Median utility
    metrics['median_utility'] = float(np.median(winner_utils))

    return metrics

# Integrate with queue processing
from agorai.queue import process_single_request

def process_with_custom_metrics(utilities, method, **params):
    """Process request with custom metrics included."""
    result = process_single_request(
        utilities,
        method=method,
        metrics=['fairness', 'efficiency', 'agreement'],
        **params
    )

    # Add custom metrics
    custom = calculate_custom_metrics(utilities, result['winner'])
    result['metrics']['custom'] = custom

    return result

# Use it
utilities = [[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]
result = process_with_custom_metrics(utilities, method="atkinson", epsilon=1.0)

print("Custom Metrics:")
print(f"  Satisfaction ratio: {result['metrics']['custom']['satisfaction_ratio']:.2%}")
print(f"  Worst-case margin: {result['metrics']['custom']['worst_case_margin']:.3f}")
```

### Metric Validation

Create validators for metric properties:

```python
def validate_fairness_metric(metric_func):
    """Validate that a metric satisfies fairness properties."""

    # Test 1: Equal utilities should give perfect fairness
    equal_utils = [[0.5, 0.5], [0.5, 0.5]]
    scores = [0.5, 0.5]
    result = metric_func(equal_utils, scores)
    assert 'gini_coefficient' in result or 'custom_metric' in result

    # Test 2: Extreme inequality should give worst fairness
    extreme_utils = [[1.0, 0.0], [0.0, 1.0]]
    scores = [0.5, 0.5]
    result = metric_func(extreme_utils, scores)

    print("✓ Metric validation passed")
    return True
```

---

## Extending the Queue System

### Custom Request Processors

Create specialized processors for your use case:

```python
from agorai.queue import load_requests_from_file, process_single_request
from typing import Dict, List, Any
import json

class CustomRequestProcessor:
    """Custom processor with preprocessing and postprocessing."""

    def __init__(self, method="atkinson", **default_params):
        self.method = method
        self.default_params = default_params
        self.results_cache = {}

    def preprocess(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess request before aggregation."""
        # Normalize utilities to [0, 1]
        utilities = item['utilities']
        normalized = []
        for agent_utils in utilities:
            min_val = min(agent_utils)
            max_val = max(agent_utils)
            if max_val > min_val:
                normalized_agent = [
                    (u - min_val) / (max_val - min_val)
                    for u in agent_utils
                ]
            else:
                normalized_agent = agent_utils
            normalized.append(normalized_agent)

        item['utilities'] = normalized
        item['original_utilities'] = utilities
        return item

    def postprocess(self, result: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess result after aggregation."""
        # Add interpretation
        winner_idx = result['winner']
        if 'candidate_labels' in item.get('metadata', {}):
            labels = item['metadata']['candidate_labels']
            result['winner_label'] = labels[winner_idx]

        # Add confidence score
        scores = result['scores']
        winner_score = scores[winner_idx]
        second_best = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
        result['confidence'] = winner_score - second_best

        return result

    def process_file(self, file_path: str, **params) -> Dict[str, Any]:
        """Process entire file with pre/post processing."""
        requests_data = load_requests_from_file(file_path)

        all_params = {**self.default_params, **params}
        results = []

        for item in requests_data['items']:
            # Preprocess
            processed_item = self.preprocess(item)

            # Aggregate
            result = process_single_request(
                processed_item['utilities'],
                method=self.method,
                metrics=['fairness', 'efficiency'],
                **all_params
            )

            # Postprocess
            result = self.postprocess(result, item)
            result['item_id'] = item.get('id', 'unknown')

            results.append(result)

        return {
            'source_file': file_path,
            'method': self.method,
            'num_requests': len(results),
            'results': results
        }

# Use custom processor
processor = CustomRequestProcessor(method="atkinson", epsilon=1.0)
results = processor.process_file("production_requests.json")

for item_result in results['results']:
    print(f"{item_result['item_id']}: {item_result.get('winner_label', 'N/A')} "
          f"(confidence: {item_result.get('confidence', 0):.2f})")
```

### Streaming Request Processing

Process large files in streaming mode:

```python
import json

def stream_process_requests(file_path: str, method: str, **params):
    """Process requests one at a time (memory efficient)."""

    with open(file_path, 'r') as f:
        data = json.load(f)

    for i, item in enumerate(data['items']):
        # Process one request
        result = process_single_request(
            item['utilities'],
            method=method,
            metrics=['fairness'],
            **params
        )

        # Yield result immediately
        yield {
            'item_id': item.get('id', f'item_{i}'),
            'winner': result['winner'],
            'scores': result['scores'],
            'metrics': result.get('metrics', {})
        }

# Use streaming processor
print("Processing requests...")
for result in stream_process_requests("large_file.json", method="majority"):
    print(f"  {result['item_id']}: Winner = {result['winner']}")
```

---

## Custom Visualization

### Creating Custom Plots

Extend visualization with your own plot types:

```python
import matplotlib.pyplot as plt
import numpy as np
from agorai.queue import process_queue

def plot_method_evolution(file_path: str, method: str, param_name: str, param_values: List):
    """Plot how results change with parameter values."""

    results_by_param = []

    for param_val in param_values:
        result = process_queue(
            file_path,
            method=method,
            metrics=['fairness', 'efficiency'],
            **{param_name: param_val}
        )
        results_by_param.append(result)

    # Extract metrics
    gini_values = [r['summary']['fairness']['gini_coefficient']
                   for r in results_by_param]
    welfare_values = [r['summary']['efficiency']['social_welfare']
                     for r in results_by_param]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(param_values, gini_values, 'o-', linewidth=2)
    ax1.set_xlabel(f'{param_name}')
    ax1.set_ylabel('Gini Coefficient')
    ax1.set_title('Fairness vs Parameter')
    ax1.grid(True, alpha=0.3)

    ax2.plot(param_values, welfare_values, 's-', linewidth=2, color='green')
    ax2.set_xlabel(f'{param_name}')
    ax2.set_ylabel('Social Welfare')
    ax2.set_title('Efficiency vs Parameter')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{method}_{param_name}_evolution.png', dpi=300)
    print(f"✓ Saved plot to {method}_{param_name}_evolution.png")

# Use it
epsilons = [0.0, 0.5, 1.0, 1.5, 2.0]
plot_method_evolution(
    "production_requests.json",
    method="atkinson",
    param_name="epsilon",
    param_values=epsilons
)
```

### Interactive Dashboards

Create interactive visualizations:

```python
def create_interactive_dashboard(file_path: str, methods: List[str]):
    """Create interactive comparison dashboard (requires plotly)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Install plotly: pip install plotly")
        return

    from agorai.queue import compare_methods_on_queue

    # Get comparison data
    comparison = compare_methods_on_queue(
        file_path,
        methods=methods,
        metrics=['fairness', 'efficiency', 'agreement']
    )

    # Extract data
    method_names = [m['method'] for m in comparison['methods']]
    gini_values = [m['summary']['fairness']['gini_coefficient']
                   for m in comparison['methods']]
    welfare_values = [m['summary']['efficiency']['social_welfare']
                     for m in comparison['methods']]
    consensus_values = [m['summary']['agreement']['consensus_score']
                       for m in comparison['methods']]

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Fairness (Gini)', 'Efficiency (Welfare)', 'Agreement (Consensus)')
    )

    # Add traces
    fig.add_trace(
        go.Bar(x=method_names, y=gini_values, name='Gini'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=method_names, y=welfare_values, name='Welfare'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=method_names, y=consensus_values, name='Consensus'),
        row=1, col=3
    )

    fig.update_layout(
        title_text="Method Comparison Dashboard",
        showlegend=False,
        height=500
    )

    fig.write_html("comparison_dashboard.html")
    print("✓ Saved interactive dashboard to comparison_dashboard.html")

# Use it
create_interactive_dashboard(
    "production_requests.json",
    methods=["majority", "atkinson", "maximin", "nash_bargaining"]
)
```

---

## Integration Patterns

### FastAPI Integration

Integrate with web APIs:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from agorai.aggregate import aggregate
from agorai.queue import process_single_request

app = FastAPI(title="AgorAI API")

class AggregationRequest(BaseModel):
    utilities: List[List[float]]
    method: str = "majority"
    params: dict = {}

class AggregationResponse(BaseModel):
    winner: int
    scores: List[float]
    method: str

@app.post("/aggregate", response_model=AggregationResponse)
def aggregate_endpoint(request: AggregationRequest):
    """Aggregation API endpoint."""
    try:
        result = aggregate(
            request.utilities,
            method=request.method,
            **request.params
        )
        return AggregationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/aggregate/metrics")
def aggregate_with_metrics(request: AggregationRequest):
    """Aggregation with metrics calculation."""
    try:
        result = process_single_request(
            request.utilities,
            method=request.method,
            metrics=['fairness', 'efficiency', 'agreement'],
            **request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn main:app --reload
```

### Django Integration

Integrate with Django:

```python
# models.py
from django.db import models
import json

class AggregationJob(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    method = models.CharField(max_length=50)
    utilities = models.JSONField()
    params = models.JSONField(default=dict)
    result = models.JSONField(null=True)
    status = models.CharField(max_length=20, default='pending')

    def process(self):
        """Process the aggregation job."""
        from agorai.aggregate import aggregate

        try:
            result = aggregate(
                self.utilities,
                method=self.method,
                **self.params
            )
            self.result = result
            self.status = 'completed'
        except Exception as e:
            self.status = 'failed'
            self.result = {'error': str(e)}

        self.save()
        return self.result

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def aggregate_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        job = AggregationJob.objects.create(
            method=data.get('method', 'majority'),
            utilities=data['utilities'],
            params=data.get('params', {})
        )

        result = job.process()

        return JsonResponse({
            'job_id': job.id,
            'result': result,
            'status': job.status
        })
```

### Celery Integration (Async Processing)

Process requests asynchronously:

```python
# tasks.py
from celery import Celery
from agorai.queue import process_queue

app = Celery('agorai_tasks', broker='redis://localhost:6379')

@app.task
def process_queue_async(file_path: str, method: str, **params):
    """Async task for queue processing."""
    result = process_queue(
        file_path,
        method=method,
        metrics=['fairness', 'efficiency', 'agreement'],
        **params
    )

    # Store result in database or cache
    return {
        'file': file_path,
        'method': method,
        'num_requests': result['num_requests'],
        'summary': result['summary']
    }

# Use it
from tasks import process_queue_async

# Submit task
task = process_queue_async.delay(
    "large_production_file.json",
    method="atkinson",
    epsilon=1.0
)

# Check status
print(f"Task ID: {task.id}")
print(f"Status: {task.status}")

# Get result when ready
if task.ready():
    result = task.get()
    print(f"Processed {result['num_requests']} requests")
```

---

## Summary

This guide covered:

✅ **Adding custom aggregation methods** - Register new methods with `register_method()`
✅ **Configuring LLM providers** - OpenAI, Anthropic, Ollama, Google
✅ **Creating custom metrics** - Extend metrics system for your needs
✅ **Extending queue system** - Custom processors and streaming
✅ **Custom visualization** - Create your own plots and dashboards
✅ **Integration patterns** - FastAPI, Django, Celery examples

For more examples, see:
- [docs/aggregate.md](aggregate.md) - Aggregation API reference
- [docs/queue.md](queue.md) - Queue processing guide
- [docs/visualization.md](visualization.md) - Visualization guide
- [examples/](../examples/) - Code examples

---

**Questions or need help?** Open an issue on GitHub or check the documentation.
