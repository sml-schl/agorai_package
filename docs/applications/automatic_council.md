# Automatic Council Creation

Automatically generate diverse perspectives for multi-agent decision-making.

## Overview

The automatic council creation module generates a set of diverse perspectives (agents) for decision-making scenarios. Instead of manually defining each agent's perspective, the system automatically creates a council of agents that represent different stakeholder viewpoints, cultural contexts, or expertise domains.

**Key Idea:** Given a decision context, automatically generate a diverse set of perspectives that are relevant and representative for that specific scenario.

## Quick Start

```python
from agorai.council import create_council, synthesize_with_council

# Automatically create a council for a hiring decision
council = create_council(
    context="hiring decision for software engineer position",
    num_perspectives=5,
    diversity_dimension="stakeholder"
)

# Use the council to make a decision
result = synthesize_with_council(
    prompt="Should we hire this candidate?",
    council=council,
    aggregation_method="fair"
)

print(f"Decision: {result['decision']}")
print(f"Perspectives: {result['perspectives_used']}")
```

---

## Main Functions

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
| `context` | `str` | *required* | Description of the decision context |
| `num_perspectives` | `int` | `5` | Number of perspectives to generate. Range: [3, 15] |
| `diversity_dimension` | `str` | `"stakeholder"` | Type of diversity to maximize. Options: `"stakeholder"`, `"cultural"`, `"expertise"`, `"demographic"`, `"ideological"` |
| `provider` | `str` | `"openai"` | LLM provider for generating perspectives |
| `model` | `str` | `"gpt-4"` | Model to use |
| `region_constraint` | `List[str]` | `None` | Constrain to specific regions (for cultural diversity) |
| `temperature` | `float` | `0.9` | Temperature for perspective generation. Range: [0.0, 2.0] |
| `ensure_minority` | `bool` | `True` | Ensure inclusion of minority/marginalized perspectives |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `perspectives` | `List[Perspective]` | List of generated perspectives |
| `diversity_score` | `float` | Measure of perspective diversity [0.0, 1.0] |
| `coverage_metrics` | `Dict` | Metrics about covered dimensions |

**Raises:**

| Exception | When |
|-----------|------|
| `ValueError` | Invalid parameters or context |
| `APIError` | LLM API errors |

---

## Diversity Dimensions

### Stakeholder Diversity

Generate perspectives representing different stakeholders affected by the decision.

**Use when:** Decisions affect multiple groups with different interests.

```python
council = create_council(
    context="implementing mandatory employee monitoring software",
    diversity_dimension="stakeholder",
    num_perspectives=5
)

# Example perspectives generated:
# - Employee privacy advocate
# - Company security officer
# - Productivity manager
# - Labor union representative
# - IT systems administrator
```

**Properties:**
- Identifies all relevant stakeholder groups
- Balances power holders and affected parties
- Includes direct and indirect stakeholders

---

### Cultural Diversity

Generate perspectives from different cultural and regional contexts.

**Use when:** Decisions have cross-cultural implications or global reach.

```python
council = create_council(
    context="content moderation policy for social media platform",
    diversity_dimension="cultural",
    num_perspectives=7
)

# Example perspectives generated:
# - Western European (GDPR-aware, privacy-focused)
# - North American (First Amendment considerations)
# - East Asian (collectivist, harmony-focused)
# - Middle Eastern (religious values)
# - Latin American (community-oriented)
# - Sub-Saharan African (Ubuntu philosophy)
# - South Asian (diverse religious contexts)
```

**Properties:**
- Represents major cultural regions
- Considers religious and philosophical traditions
- Accounts for legal and regulatory contexts

---

### Expertise Diversity

Generate perspectives from different domains of expertise.

**Use when:** Decisions require interdisciplinary knowledge.

```python
council = create_council(
    context="deploying AI system for medical diagnosis",
    diversity_dimension="expertise",
    num_perspectives=6
)

# Example perspectives generated:
# - Medical doctor (clinical expertise)
# - Machine learning engineer (technical implementation)
# - Bioethicist (ethical implications)
# - Patient advocate (user experience)
# - Healthcare administrator (operational feasibility)
# - Medical liability lawyer (legal risks)
```

**Properties:**
- Covers relevant knowledge domains
- Balances technical and human perspectives
- Includes domain-specific expertise

---

### Demographic Diversity

Generate perspectives representing different demographic groups.

**Use when:** Decisions may have differential impact across demographics.

```python
council = create_council(
    context="designing public transportation system",
    diversity_dimension="demographic",
    num_perspectives=6
)

# Example perspectives generated:
# - Elderly person with mobility challenges
# - Working parent with young children
# - Low-income commuter
# - Person with visual impairment
# - Teenage student
# - Rural area resident
```

**Properties:**
- Represents protected classes (age, disability, income)
- Ensures marginalized groups are included
- Considers intersectional identities

---

### Ideological Diversity

Generate perspectives from different political and ideological viewpoints.

**Use when:** Decisions involve contested values or political trade-offs.

```python
council = create_council(
    context="government data collection for public health",
    diversity_dimension="ideological",
    num_perspectives=5
)

# Example perspectives generated:
# - Civil libertarian (individual rights)
# - Public health utilitarian (collective welfare)
# - Conservative (limited government)
# - Progressive (social justice)
# - Technocratic pragmatist (evidence-based)
```

**Properties:**
- Spans political spectrum
- Represents core value systems
- Avoids stereotyping while capturing genuine differences

---

## Advanced Usage

### Custom Perspective Templates

Define custom templates for generating perspectives:

```python
from agorai.council import create_council, PerspectiveTemplate

# Define custom templates
templates = [
    PerspectiveTemplate(
        name="budget_conscious",
        system_prompt="You prioritize cost-effectiveness and fiscal responsibility...",
        values=["efficiency", "accountability"]
    ),
    PerspectiveTemplate(
        name="innovation_focused",
        system_prompt="You prioritize innovation and competitive advantage...",
        values=["growth", "disruption"]
    ),
    PerspectiveTemplate(
        name="risk_averse",
        system_prompt="You prioritize safety and risk mitigation...",
        values=["stability", "security"]
    )
]

council = create_council(
    context="adopting new technology platform",
    custom_templates=templates
)
```

---

### Hybrid Councils

Combine automatic generation with manually defined perspectives:

```python
from agorai.council import create_council, Perspective

# Define some perspectives manually
manual_perspectives = [
    Perspective(
        name="ceo",
        system_prompt="As CEO, you prioritize shareholder value and company reputation...",
        provider="openai",
        model="gpt-4"
    ),
    Perspective(
        name="lead_engineer",
        system_prompt="As lead engineer, you prioritize technical excellence and maintainability...",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )
]

# Automatically generate additional perspectives
council = create_council(
    context="deciding on technical architecture",
    num_perspectives=3,  # Generate 3 more
    diversity_dimension="stakeholder",
    initial_perspectives=manual_perspectives  # Start with manual ones
)

# Final council has 5 perspectives (2 manual + 3 generated)
```

---

### Diversity Maximization

Ensure maximum diversity using multiple dimensions:

```python
council = create_council(
    context="content moderation policy",
    num_perspectives=10,
    diversity_dimensions=["cultural", "demographic", "ideological"],  # Multiple dimensions
    diversity_optimization="maximize",
    ensure_minority=True
)

print(f"Diversity score: {council.diversity_score:.2f}")
print(f"Coverage: {council.coverage_metrics}")
```

---

## Council Composition Strategies

### Balanced Representation

Equal weight to all perspectives:

```python
council = create_council(
    context="...",
    composition_strategy="balanced",
    num_perspectives=6
)

# All perspectives have equal influence in aggregation
```

---

### Minority-Amplified

Over-represent marginalized perspectives:

```python
council = create_council(
    context="...",
    composition_strategy="minority_amplified",
    num_perspectives=6,
    minority_multiplier=2.0  # Minority perspectives count 2x
)

# Perspectives from marginalized groups have greater weight
```

---

### Expertise-Weighted

Weight perspectives by relevant expertise:

```python
council = create_council(
    context="technical security decision",
    composition_strategy="expertise_weighted",
    num_perspectives=6,
    expertise_domain="cybersecurity"
)

# Cybersecurity experts have greater weight than others
```

---

## Using Councils for Synthesis

### Basic Synthesis

```python
from agorai.council import create_council, synthesize_with_council

council = create_council(
    context="hiring decision",
    num_perspectives=5
)

result = synthesize_with_council(
    prompt="Should we hire candidate X?",
    council=council,
    aggregation_method="fair"
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")

# View individual perspectives
for perspective_result in result['individual_results']:
    print(f"{perspective_result['perspective']}: {perspective_result['opinion']}")
```

---

### With Different Aggregation Methods

```python
# Compare different aggregation strategies
methods = ["democratic", "fair", "minority-focused", "robust"]

for method in methods:
    result = synthesize_with_council(
        prompt="Should we implement this policy?",
        council=council,
        aggregation_method=method
    )
    print(f"{method}: {result['decision']} ({result['confidence']:.0%})")
```

---

## Best Practices

### 1. Context Specificity

**Good context:**
```python
council = create_council(
    context="content moderation decision for political speech on social media platform with global user base",
    diversity_dimension="cultural"
)
```

**Poor context:**
```python
council = create_council(
    context="content moderation",  # Too vague
    diversity_dimension="cultural"
)
```

---

### 2. Appropriate Diversity Dimension

| Decision Type | Recommended Dimension |
|---------------|----------------------|
| Business strategy | Stakeholder |
| Global product | Cultural |
| Technical architecture | Expertise |
| Social policy | Demographic + Ideological |
| Content moderation | Cultural + Demographic |

---

### 3. Council Size

| Num Perspectives | Use When | Trade-off |
|------------------|----------|-----------|
| 3-5 | Quick decisions, low stakes | Fast, less diverse |
| 5-7 | Standard use cases | Balanced |
| 8-12 | High-stakes, complex decisions | Slow, more comprehensive |
| 12+ | Critical decisions requiring maximum legitimacy | Very slow, diminishing returns |

---

### 4. Validate Diversity

```python
council = create_council(context="...", num_perspectives=7)

# Check diversity metrics
print(f"Diversity score: {council.diversity_score:.2f}")
print(f"Dimensions covered: {council.coverage_metrics['dimensions']}")
print(f"Unique viewpoints: {council.coverage_metrics['unique_viewpoints']}")

# If diversity is too low, regenerate
if council.diversity_score < 0.6:
    council = create_council(
        context="...",
        num_perspectives=7,
        temperature=1.0,  # Increase randomness
        diversity_optimization="maximize"
    )
```

---

## Limitations

### 1. Prompt-Based Perspectives

Current implementation uses prompt engineering to generate perspectives, which:
- ✅ Fast and flexible
- ✅ Can represent any perspective
- ❌ May rely on stereotypes
- ❌ Limited by base model's training

**Mitigation:** See [Bias Mitigation](bias_mitigation.md) for model-based approaches.

---

### 2. Perspective Authenticity

Generated perspectives approximate but don't fully capture:
- Lived experiences of marginalized groups
- Deep cultural knowledge
- Domain expertise developed over years

**Mitigation:** Validate critical decisions with actual stakeholders.

---

### 3. Coverage Gaps

Automatic generation may miss:
- Niche stakeholder groups
- Emerging perspectives
- Context-specific concerns

**Mitigation:** Use hybrid councils with manual perspective definitions.

---

## Examples

### Example 1: Hiring Decision

```python
from agorai.council import create_council, synthesize_with_council

council = create_council(
    context="hiring for senior engineering role at fintech startup",
    diversity_dimension="stakeholder",
    num_perspectives=6
)

result = synthesize_with_council(
    prompt="""
    Candidate Profile:
    - 10 years experience
    - Strong technical skills
    - No formal computer science degree (self-taught)
    - Previously worked at BigTech companies
    - Requesting high salary

    Should we hire this candidate?
    """,
    council=council,
    aggregation_method="fair"
)

print(f"Decision: {result['decision']}")
print("\nPerspectives:")
for p in result['individual_results']:
    print(f"  {p['perspective']}: {p['opinion']} (confidence: {p['confidence']:.0%})")
```

---

### Example 2: Content Moderation

```python
council = create_council(
    context="moderating political content on global social media platform",
    diversity_dimension="cultural",
    num_perspectives=8,
    ensure_minority=True
)

result = synthesize_with_council(
    prompt="Should this post be removed: 'Citizens should protest peacefully against government policy'?",
    council=council,
    aggregation_method="minority-focused"  # Protect minority perspectives
)

print(f"Decision: {result['decision']}")
print(f"Cultural perspectives: {result['perspectives_used']}")
```

---

### Example 3: Product Feature Decision

```python
council = create_council(
    context="adding AI-powered personalization to education app used by children",
    diversity_dimensions=["expertise", "stakeholder"],
    num_perspectives=8
)

result = synthesize_with_council(
    prompt="Should we implement AI-powered personalized learning paths?",
    council=council,
    aggregation_method="robust"  # Outlier-resistant
)

# Check for dissenting opinions
minority_opinions = [p for p in result['individual_results'] if p['opinion'] != result['decision']]
print(f"Minority opinions: {len(minority_opinions)}")
for opinion in minority_opinions:
    print(f"  {opinion['perspective']}: {opinion['rationale']}")
```

---

## See Also

- [Bias Mitigation](bias_mitigation.md) - Use councils for bias detection
- [Aggregation Methods](../core/aggregation.md) - Choose aggregation strategy
- [Property Analysis](../core/properties.md) - Theoretical guarantees
- [API Reference](../reference/api.md) - Complete function signatures
