# Automated Counterfactual Testing for Causal Robustness Evaluation

## Overview

The automated counterfactual testing module provides a complete pipeline for evaluating whether AI bias detection systems make decisions for the **right reasons** (causal) rather than relying on **spurious correlations**.

### The Problem

Traditional evaluation compares model predictions to labels:
- **Accuracy = 85%** looks good ✓
- But what if the model achieves 85% by learning "images with kittens are harmless" instead of understanding the actual content?

This is a **spurious correlation** - a pattern that correlates with the label but isn't causally relevant to the task.

### The Solution

Counterfactual testing systematically modifies **protected attributes** (ethnicity, gender, age, religious symbols, etc.) that should be causally irrelevant to hate speech detection. If changing these attributes causes prediction changes, it reveals spurious correlations.

**Example:**
- Original: [Image of person with hijab] + "Have a nice day" → **Not Hateful** ✓
- Counterfactual: [Same image, hijab → cross] + "Have a nice day" → **Hateful** ✗

This inconsistency reveals the model is using religious symbols as a shortcut rather than understanding the message.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Automated Counterfactual Testing Pipeline       │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼────────────┐      ┌──────────▼─────────────┐
        │  Protected Attribute    │      │  Counterfactual        │
        │  Detector               │      │  Generator             │
        │                         │      │                        │
        │  Uses: GPT-4V, Claude   │      │  Uses: DALL-E 3,       │
        │  Output: Detected attrs │      │  Stable Diffusion      │
        └───────────┬─────────────┘      │  Output: Modified imgs │
                    │                    └──────────┬─────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  Consistency           │
                        │  Evaluator             │
                        │                        │
                        │  Compare predictions   │
                        │  Identify spurious     │
                        │  correlations          │
                        └────────────────────────┘
```

---

## Components

### 1. Protected Attribute Detector

**Purpose:** Identify protected attributes in images using Vision-Language Models.

**Detected Attributes:**
- **Demographics:** Ethnicity, gender, age, religion
- **Visual markers:** Skin tone, clothing style, religious symbols, cultural symbols
- **Contextual:** Socioeconomic indicators, ability status, body type
- **Objects/Background:** Cultural objects, geographic indicators

**Implementation:** `src/agorai/testing/protected_attribute_detector.py`

**Example:**
```python
from agorai.testing import ProtectedAttributeDetector, create_detector_from_config

detector = create_detector_from_config({
    "vlm_model": "gpt-4-vision-preview",
    "confidence_threshold": 0.6
})

result = detector.detect_from_url(
    image_url="https://example.com/image.jpg",
    text_content="Meme caption"
)

for attr in result.detected_attributes:
    print(f"{attr.attribute_type}: {attr.value} (confidence: {attr.confidence})")
```

---

### 2. Counterfactual Generator

**Purpose:** Generate counterfactual images by modifying protected attributes.

**Supported Tools:**
- **DALL-E 3** (OpenAI) - High quality, text-only prompts
- **Stable Diffusion XL** (Replicate/HuggingFace) - Image-to-image editing
- **Replicate API** - Multiple models
- **Local Diffusion** - Self-hosted

**Modification Strategies:**
- **SWAP:** Change attribute to specific alternative (e.g., male → female)
- **REMOVE:** Remove attribute entirely
- **NEUTRALIZE:** Make attribute ambiguous
- **VARY_SPECTRUM:** Vary along continuum (e.g., skin tone)
- **CROSS_CULTURAL:** Swap to different cultural context

**Implementation:** `src/agorai/testing/counterfactual_generator.py`

**Example:**
```python
from agorai.testing import CounterfactualGenerator, create_generator_from_config

generator = create_generator_from_config({
    "tool": "dall-e-3",
    "api_key_env": "OPENAI_API_KEY"
})

# Automatically generate variations for detected attributes
counterfactuals = generate_all_variations(
    generator=generator,
    original_image_url="https://example.com/image.jpg",
    detected_attributes=detected_attrs,
    max_variations_per_attribute=2
)
```

---

### 3. Automated Testing Pipeline

**Purpose:** Complete end-to-end pipeline from detection to evaluation.

**Pipeline Steps:**
1. **Detect** protected attributes in original image
2. **Generate** counterfactual variations (modify attributes)
3. **Evaluate** model predictions on original + counterfactuals
4. **Compute** consistency metrics
5. **Identify** spurious correlations

**Implementation:** `src/agorai/testing/automated_counterfactual_testing.py`

**Example:**
```python
from agorai.testing import create_tester_from_config
from agorai import AgentCouncil

# Create your bias detection system
council = AgentCouncil(agents=[...])

# Create automated tester
tester = create_tester_from_config(council, {
    "detector": {"vlm_model": "gpt-4-vision-preview"},
    "generator": {"tool": "dall-e-3"},
    "testing": {"max_attributes_to_test": 3}
})

# Run test
result = tester.run_test_case(
    test_case_id="test_001",
    image_url="https://example.com/meme.jpg",
    text_content="Caption"
)

print(f"Consistency score: {result.consistency_score:.2%}")
print(f"Spurious correlations: {result.spurious_correlation_detected}")
```

---

## Consistency Metrics

### Counterfactual Consistency (CC)

Measures how often the model's prediction remains unchanged when irrelevant attributes are modified.

**Formula:**
```
CC = (# consistent predictions) / (# total counterfactuals)
```

**Interpretation:**
- **CC ≥ 90%:** Strong causal reasoning ✓
- **CC = 70-90%:** Moderate robustness ⚠️
- **CC < 70%:** Likely spurious correlations ✗

### Spurious Correlation Detection

A spurious correlation is detected when:
1. Modifying a protected attribute causes prediction to change
2. The attribute should be causally irrelevant (e.g., ethnicity in hate speech detection)
3. The change is consistent across multiple variations

**Example Output:**
```
Spurious correlations detected:
  - ethnicity: 3 occurrences, avg confidence delta: 0.35
    → Model likely using ethnicity as shortcut
  - religious_symbols: 2 occurrences, avg confidence delta: 0.28
    → Model likely relying on religious markers
```

---

## Configuration

**File:** `config/counterfactual_testing_config.yaml`

```yaml
# Detector settings
detector:
  vlm_model: "gpt-4-vision-preview"
  confidence_threshold: 0.6

# Generator settings
generator:
  tool: "dall-e-3"  # or "stable-diffusion-xl"
  provider: "openai"

# Testing settings
testing:
  max_attributes_to_test: 3
  max_variations_per_attribute: 2
  consistency_threshold: 0.8
  significant_confidence_delta: 0.2

# Output settings
output:
  results_directory: "./counterfactual_test_results"
  save_counterfactual_images: true
```

---

## Usage Examples

### Example 1: Single Image Test

```python
from agorai.testing import AutomatedCounterfactualTester
from agorai import AgentCouncil

# Create council
council = AgentCouncil(agents=[western_agent, eastern_agent, global_south_agent])

# Create tester
tester = AutomatedCounterfactualTester(
    council=council,
    detector=detector,
    generator=generator,
    config=testing_config
)

# Run test
result = tester.run_test_case(
    test_case_id="hateful_memes_001",
    image_url="https://example.com/meme.jpg",
    text_content="Caption text",
    ground_truth_label="hateful"
)

# Results
print(f"Protected attributes: {len(result.protected_attributes)}")
print(f"Counterfactuals generated: {len(result.counterfactuals)}")
print(f"Consistency score: {result.consistency_score:.2%}")

if result.spurious_correlation_detected:
    print("⚠️  Spurious correlations found - investigate model reasoning!")
```

### Example 2: Batch Testing

```python
test_cases = [
    {
        "id": "meme_001",
        "image_url": "https://example.com/meme_001.jpg",
        "text_content": "Text 1",
        "ground_truth": "hateful"
    },
    {
        "id": "meme_002",
        "image_url": "https://example.com/meme_002.jpg",
        "text_content": "Text 2",
        "ground_truth": "not_hateful"
    },
    # ... more cases
]

results = tester.run_batch_tests(test_cases)

# Aggregate statistics
avg_consistency = sum(r.consistency_score for r in results) / len(results)
spurious_count = sum(1 for r in results if r.spurious_correlation_detected)

print(f"Average consistency: {avg_consistency:.2%}")
print(f"Tests with spurious correlations: {spurious_count}/{len(results)}")

# Export results
tester.export_results(results, "results.json")
```

### Example 3: Custom Variations

```python
from agorai.testing import (
    ProtectedAttribute,
    ProtectedAttributeType,
    create_counterfactual_variation,
    ModificationStrategy
)

# Define custom attribute
attr = ProtectedAttribute(
    attribute_type=ProtectedAttributeType.GENDER,
    value="female",
    confidence=0.95
)

# Create variations
variations = [
    create_counterfactual_variation(
        attribute=attr,
        target_value="male",
        strategy=ModificationStrategy.SWAP,
        preserve_elements=["text", "background", "pose"]
    ),
    create_counterfactual_variation(
        attribute=attr,
        target_value="non-binary",
        strategy=ModificationStrategy.SWAP
    )
]

# Generate using these specific variations
request = CounterfactualGenerationRequest(
    original_image_url=image_url,
    variations=variations,
    generation_tool=ImageGenerationTool.DALL_E_3
)

result = generator.generate(request)
```

---

## Integration with Thesis Evaluation

### Chapter 7: Evaluation

Use automated counterfactual testing to demonstrate:

1. **Traditional Metrics** (baseline)
   - Accuracy, precision, recall, F1 on Hateful Memes

2. **Causal Robustness Metrics** (novel contribution)
   - Counterfactual consistency scores
   - Spurious correlation detection
   - Edge case robustness

3. **Mechanism Comparison**
   - Which aggregation mechanisms achieve best causal robustness?
   - Do theoretical properties (Condorcet, Pareto) predict robustness?

**Example Results Table:**

| Mechanism | Traditional Acc | CF Consistency | Spurious Corr Count | Causal Grade |
|-----------|----------------|----------------|---------------------|--------------|
| Majority  | 85%            | 75%            | 18                  | C            |
| Maximin   | 88%            | 92%            | 7                   | A            |
| Atkinson  | 86%            | 90%            | 9                   | A-           |
| Schulze   | 89%            | 88%            | 11                  | B+           |

**Key Finding:** Fairness-focused mechanisms (maximin, Atkinson) achieve higher causal robustness than efficiency-focused ones (majority), despite similar traditional accuracy.

---

## API Reference

### ProtectedAttributeDetector

```python
class ProtectedAttributeDetector:
    def __init__(
        self,
        vlm_agent: BaseAgent,
        default_confidence_threshold: float = 0.5
    )

    def detect(
        self,
        request: ProtectedAttributeDetectionRequest
    ) -> ProtectedAttributeDetectionResponse

    def detect_from_url(
        self,
        image_url: str,
        text_content: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> ProtectedAttributeDetectionResponse
```

### CounterfactualGenerator

```python
class CounterfactualGenerator:
    def __init__(
        self,
        image_gen_api: ImageGenerationAPI,
        default_tool: ImageGenerationTool = ImageGenerationTool.DALL_E_3
    )

    def generate(
        self,
        request: CounterfactualGenerationRequest
    ) -> CounterfactualGenerationResponse
```

### AutomatedCounterfactualTester

```python
class AutomatedCounterfactualTester:
    def __init__(
        self,
        council: AgentCouncil,
        detector: ProtectedAttributeDetector,
        generator: CounterfactualGenerator,
        config: AutomatedTestingConfig
    )

    def run_test_case(
        self,
        test_case_id: str,
        image_url: str,
        text_content: Optional[str] = None,
        ground_truth_label: Optional[str] = None
    ) -> CounterfactualTestCase

    def run_batch_tests(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[CounterfactualTestCase]

    def export_results(
        self,
        test_cases: List[CounterfactualTestCase],
        output_path: str
    )
```

---

## Limitations

### 1. Image Generation Quality

Counterfactual images may not perfectly preserve all aspects:
- Composition might shift slightly
- Style may vary
- Background elements might change

**Mitigation:** Use `preserve_composition`, `preserve_style`, `preserve_background` flags.

### 2. API Costs

Generating counterfactuals requires:
- VLM API calls for detection (e.g., GPT-4V)
- Image generation API calls (e.g., DALL-E 3)

**Cost estimate:** ~$0.50-1.00 per test case with 3 attributes × 2 variations

**Mitigation:**
- Use batch processing
- Limit `max_variations_per_attribute`
- Cache results

### 3. Protected Attribute Detection Accuracy

VLMs may:
- Miss subtle attributes
- Misidentify attributes
- Have varying confidence

**Mitigation:**
- Set appropriate `confidence_threshold`
- Manual validation of detected attributes
- Use multiple detection rounds

### 4. Counterfactual Validity

Generated counterfactuals might:
- Introduce unintended changes
- Not be photorealistic
- Alter semantically relevant features

**Mitigation:**
- Human review of generated images
- Quality filtering
- Document limitations in thesis

---

## Future Enhancements

1. **Improved Image Editing**
   - Use InstructPix2Pix for more precise edits
   - Implement mask-based editing for targeted changes
   - Support for video counterfactuals

2. **Automatic Attribute Mapping**
   - Learn which attributes are causally relevant for each task
   - Adaptive counterfactual generation based on task

3. **Multilingual Support**
   - Extend to non-English text
   - Cross-lingual counterfactuals

4. **Interactive UI**
   - Visual dashboard for exploring counterfactuals
   - Manual counterfactual creation interface
   - Real-time consistency monitoring

---

## References

This implementation is based on:

- **Pearl's Causal Hierarchy:** Three levels of causal reasoning (association, intervention, counterfactual)
- **Fairness Through Awareness:** Dwork et al. (2012) - Similar individuals should be treated similarly
- **Counterfactual Fairness:** Kusner et al. (2017) - Predictions should be consistent under protected attribute interventions

**For thesis citations:**
```latex
@article{pearl2009causal,
  title={Causal inference in statistics: An overview},
  author={Pearl, Judea},
  journal={Statistics surveys},
  volume={3},
  pages={96--146},
  year={2009}
}

@article{kusner2017counterfactual,
  title={Counterfactual fairness},
  author={Kusner, Matt J and Loftus, Joshua and Russell, Chris and Silva, Ricardo},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

---

## Support

For issues or questions:
- Check demo script: `examples/automated_counterfactual_testing_demo.py`
- Review configuration: `config/counterfactual_testing_config.yaml`
- See implementation: `src/agorai/testing/`

**Common Issues:**
1. **API rate limits:** Add delays or reduce parallel tests
2. **Generation failures:** Check image URL accessibility, API credits
3. **Inconsistent results:** Increase `confidence_threshold`, review prompts
