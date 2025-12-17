# Bias Mitigation

Detect and mitigate AI bias through multi-perspective analysis.

## Overview

The bias mitigation module synthesizes diverse cultural and demographic perspectives to identify and reduce bias in AI decision-making.

**Key Idea:** Instead of trusting a single AI model's judgment on sensitive content, gather perspectives from multiple cultural contexts and aggregate them democratically.

## Quick Start

```python
from agorai.bias import mitigate_bias

result = mitigate_bias(
    input_text="Is this hiring decision fair?",
    aggregation_method="fair",
    num_perspectives=5
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Fairness: {result['fairness_metrics']}")
```

---

## Main Function

### `mitigate_bias()`

Analyze content from multiple cultural perspectives and aggregate results.

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
| `input_image` | `str` | `None` | Optional image URL or base64 for multimodal analysis |
| `aggregation_method` | `str` | `"fair"` | Method for aggregating perspectives. See [aggregation methods](../core/aggregation.md) |
| `num_perspectives` | `int` | `5` | Number of cultural perspectives to generate. Range: [3, 10] |
| `provider` | `str` | `"openai"` | LLM provider. Options: `"openai"`, `"anthropic"`, `"ollama"` |
| `model` | `str` | `"gpt-4"` | Model to use. Provider-specific |
| `cultural_regions` | `List[str]` | `None` | Specific regions to include. If `None`, auto-selects diverse regions |
| `temperature` | `float` | `0.7` | LLM temperature for generation. Range: [0.0, 2.0] |
| `return_individual` | `bool` | `False` | Whether to return individual perspective results |
| `fairness_metrics` | `List[str]` | `["demographic_parity", "equalized_odds"]` | Metrics to compute |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `str` | Aggregated decision (e.g., "biased", "not_biased", "uncertain") |
| `confidence` | `float` | Confidence score [0.0, 1.0] |
| `fairness_metrics` | `Dict` | Computed fairness metrics |
| `perspectives_used` | `List[str]` | Cultural regions included in analysis |
| `aggregation_metadata` | `Dict` | Details of aggregation process |
| `individual_results` | `List[Dict]` | Individual perspective results (if `return_individual=True`) |

**Raises:**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid parameters, unsupported provider |
| `APIError` | LLM API errors |

---

## Cultural Regions

### Predefined Regions

The system can automatically select from these predefined cultural regions:

| Region Code | Description | Typical Values |
|-------------|-------------|----------------|
| `western_europe` | Western European perspectives | Privacy-focused, GDPR-aware |
| `north_america` | North American (US/Canada) | First Amendment considerations |
| `east_asia` | East Asian perspectives | Collectivist, harmony-focused |
| `south_asia` | South Asian perspectives | Diverse religious contexts |
| `middle_east` | Middle Eastern perspectives | Islamic values, honor culture |
| `latin_america` | Latin American perspectives | Community-oriented |
| `sub_saharan_africa` | Sub-Saharan African perspectives | Ubuntu philosophy |
| `oceania` | Australian/NZ perspectives | Indigenous considerations |

### Auto-Selection

If `cultural_regions=None` (default), the system automatically selects diverse regions:

```python
result = mitigate_bias(
    input_text="...",
    num_perspectives=5
)
# Automatically selects 5 diverse regions
# Example: [western_europe, east_asia, north_america, south_asia, latin_america]
```

### Manual Selection

Specify exactly which regions to include:

```python
result = mitigate_bias(
    input_text="...",
    cultural_regions=["western_europe", "middle_east", "east_asia"]
)
# Uses exactly these 3 regions (ignores num_perspectives)
```

---

## Multimodal Analysis

### Text + Image

Analyze both text and image together (requires vision-capable model):

```python
result = mitigate_bias(
    input_text="Caption for this meme",
    input_image="https://example.com/meme.jpg",
    provider="openai",
    model="gpt-4-vision-preview"
)
```

### Image Only

Analyze image without text:

```python
result = mitigate_bias(
    input_text="",  # Empty text
    input_image="https://example.com/image.jpg",
    model="gpt-4-vision-preview"
)
```

### Supported Image Formats

- **URL**: `"https://example.com/image.jpg"`
- **Base64**: `"data:image/jpeg;base64,/9j/4AAQ..."`
- **Local file**: `"file:///path/to/image.jpg"` (converted to base64 automatically)

---

## Fairness Metrics

### Available Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `demographic_parity` | Equal decision rates across groups | Gap should be close to 0 |
| `equalized_odds` | Equal TPR/FPR across groups | Both gaps should be close to 0 |
| `worst_group_accuracy` | Accuracy on worst-performing group | Higher is better |
| `disparate_impact` | Ratio of positive rates | Should be close to 1.0 |

### Specifying Metrics

```python
result = mitigate_bias(
    input_text="...",
    fairness_metrics=["demographic_parity", "equalized_odds", "worst_group_accuracy"]
)

print(result['fairness_metrics'])
# {
#     'demographic_parity_gap': 0.12,
#     'equalized_odds_gap': 0.08,
#     'worst_group_accuracy': 0.85
# }
```

---

## Examples

### Example 1: Content Moderation

```python
from agorai.bias import mitigate_bias

# Analyze potentially offensive content
result = mitigate_bias(
    input_text="Is this comment hate speech: 'Those people are all the same'",
    aggregation_method="fair",
    num_perspectives=7
)

if result['decision'] == 'biased':
    print(f"Content flagged as potentially biased (confidence: {result['confidence']:.0%})")
    print(f"Perspectives that flagged it: {result['flagged_by']}")
```

### Example 2: Hiring Decision Analysis

```python
result = mitigate_bias(
    input_text="""
    Resume Review:
    - Candidate has 10 years experience
    - Graduated from community college
    - Name suggests Middle Eastern background
    - Strong technical skills
    """,
    aggregation_method="minority-focused",  # Protect against discrimination
    cultural_regions=["north_america", "middle_east", "western_europe"]
)

print(f"Decision: {result['decision']}")
print(f"Demographic parity: {result['fairness_metrics']['demographic_parity_gap']:.3f}")
```

### Example 3: Multimodal Meme Analysis

```python
result = mitigate_bias(
    input_text="When you see your ex at the party",
    input_image="https://example.com/meme.jpg",
    aggregation_method="robust",  # Resist outlier opinions
    model="gpt-4-vision-preview",
    num_perspectives=5,
    return_individual=True
)

# See how different cultures interpret the meme
for perspective in result['individual_results']:
    print(f"{perspective['region']}: {perspective['interpretation']}")
```

### Example 4: Comparison Across Methods

```python
input_text = "Gender-specific job posting"

methods = ["democratic", "fair", "minority-focused", "robust"]
for method in methods:
    result = mitigate_bias(input_text, aggregation_method=method)
    print(f"{method}: {result['decision']} ({result['confidence']:.0%})")

# Output might show:
# democratic: not_biased (65%)
# fair: uncertain (52%)
# minority-focused: biased (73%)
# robust: biased (68%)
```

---

## Advanced Usage

### Custom Perspectives

Instead of using predefined regions, define custom perspectives:

```python
from agorai.bias import BiasConfig, mitigate_bias_custom

config = BiasConfig(
    perspectives=[
        {
            "name": "conservative_perspective",
            "system_prompt": "Analyze from a conservative viewpoint...",
            "provider": "openai",
            "model": "gpt-4"
        },
        {
            "name": "progressive_perspective",
            "system_prompt": "Analyze from a progressive viewpoint...",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022"
        },
        {
            "name": "libertarian_perspective",
            "system_prompt": "Analyze from a libertarian viewpoint...",
            "provider": "ollama",
            "model": "llama3.2"
        }
    ],
    aggregation_method="fair"
)

result = mitigate_bias_custom(input_text="...", config=config)
```

### Batch Processing

Process multiple inputs:

```python
from agorai.bias import mitigate_bias_batch

inputs = [
    {"text": "Input 1", "image": None},
    {"text": "Input 2", "image": "url1.jpg"},
    {"text": "Input 3", "image": "url2.jpg"},
]

results = mitigate_bias_batch(
    inputs,
    aggregation_method="fair",
    num_perspectives=5
)

for i, result in enumerate(results):
    print(f"Input {i+1}: {result['decision']}")
```

---

## Best Practices

### 1. Choose Appropriate Number of Perspectives

| Num Perspectives | Use When | Trade-off |
|------------------|----------|-----------|
| 3 | Quick checks, low stakes | Faster, less diverse |
| 5 | Most common use cases | Balanced |
| 7+ | High-stakes decisions | Slower, more robust |

### 2. Select Aggregation Method Carefully

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Equal treatment | `fair` (Atkinson) | Balances all perspectives |
| Protect minorities | `minority-focused` (Maximin) | Ensures no group harmed |
| Resist manipulation | `robust` (Robust Median) | Outlier-resistant |
| Democratic legitimacy | `democratic` (Majority) | Simple majority rule |

### 3. Consider Context

```python
# For legal compliance (strict)
result = mitigate_bias(
    input_text="...",
    aggregation_method="minority-focused",  # No group should be harmed
    num_perspectives=7  # More perspectives for robustness
)

# For content recommendations (lenient)
result = mitigate_bias(
    input_text="...",
    aggregation_method="democratic",  # Simple majority
    num_perspectives=3  # Fewer perspectives for speed
)
```

### 4. Interpret Confidence Scores

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| > 0.8 | High agreement | Trust decision |
| 0.5-0.8 | Moderate agreement | Review carefully |
| < 0.5 | Low agreement | Human review required |

---

## Limitations

### 1. Prompt-Based Perspectives

Current implementation uses prompt-based cultural perspectives, which:
- ✅ Are fast and flexible
- ✅ Can represent any culture/demographic
- ❌ May rely on stereotypes
- ❌ Are limited by base model's training

**Mitigation:** See [automatic council creation](automatic_council.md) for model-based perspectives.

### 2. Ground Truth Challenge

"Bias" is subjective and context-dependent:
- No objective ground truth for what is "biased"
- Different cultures have different norms
- Legal definitions vary by jurisdiction

**Mitigation:** Use multiple fairness metrics, transparent decision process.

### 3. Computational Cost

Multiple LLM calls can be expensive:
- 5 perspectives = 5× API cost
- Vision models are more expensive than text-only

**Mitigation:** Cache results, use batch processing, adjust `num_perspectives` based on stakes.

---

## See Also

- [Aggregation Methods](../core/aggregation.md) - Choose the right aggregation method
- [Automatic Council Creation](automatic_council.md) - Auto-generate perspectives
- [Counterfactual Testing](counterfactual_testing.md) - Test causal robustness
- [API Reference](../reference/api.md) - Complete function signatures
