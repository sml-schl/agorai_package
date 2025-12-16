# Structured Opinion Synthesis

## Overview

The structured synthesis feature in AgorAI ensures that LLM agents provide properly formatted responses with clear option selection and reasoning. This is critical for robust mathematical aggregation in democratic AI systems.

## Key Features

### 1. **Structured Response Format**

All agents must respond in a standardized format:

```json
{
    "response": <number>,
    "reasoning": "<explanation>"
}
```

- **`response`**: The chosen option number (1-based indexing)
- **`reasoning`**: Brief explanation for the choice

### 2. **Automatic Validation**

Responses are validated using multiple strategies:
- **JSON pattern matching**: Detects properly formatted JSON responses
- **Colon-separated format**: Supports `response: N, reasoning: ...` format
- **Lenient parsing**: Optionally extracts first number from informal text
- **Range checking**: Ensures response number is within valid option range

### 3. **Retry Mechanism**

When an agent provides an invalid response:
1. The system detects the format violation
2. A correction prompt is automatically generated
3. The agent is given another chance (up to `max_retries`)
4. The retry prompt explicitly states the required format

### 4. **Clear Interfaces**

Well-defined TypedDicts and return types ensure:
- Type safety throughout the pipeline
- Clear input/output contracts
- Easy debugging and testing

## Usage

### Basic Example

```python
from agorai import Agent, synthesize_structured

# Define your question and options
question = "Should we approve this feature?"
options = [
    "Approve immediately",
    "Reject",
    "Request more information"
]

# Create agents
agents = [
    Agent("openai", "gpt-4", api_key="sk-..."),
    Agent("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
    Agent("ollama", "llama3.2")
]

# Get structured decision
result = synthesize_structured(
    question=question,
    options=options,
    agents=agents,
    aggregation_method="majority"
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")

# Access individual agent reasoning
for agent_output in result['agent_outputs']:
    print(f"{agent_output['agent']}: Option {agent_output['response_number']}")
    print(f"  Reasoning: {agent_output['reasoning']}")
```

### With Additional Context

```python
result = synthesize_structured(
    question="Which technology should we use?",
    options=["Python", "Java", "Go"],
    agents=agents,
    context="Budget: $50k, Timeline: 3 months, Team size: 5",
    aggregation_method="atkinson",
    epsilon=0.8
)
```

### Using Council Directly

```python
from agorai import Agent, Council

agents = [
    Agent("ollama", "llama3.2", name="Agent 1", max_retries=3),
    Agent("ollama", "llama3.2", name="Agent 2", max_retries=3)
]

council = Council(agents, aggregation_method="majority")

result = council.decide_structured(
    question="Is this ready for production?",
    options=["Yes", "No", "Needs more testing"],
    strict=True  # Enforce strict format validation
)
```

### Custom Agent Configuration

```python
# Configure retry behavior per agent
agent = Agent(
    provider="openai",
    model="gpt-4",
    api_key="sk-...",
    max_retries=2,  # Allow 2 retries for formatting corrections
    temperature=0.7,
    name="Technical Reviewer"
)

# Generate structured response directly
response = agent.generate_structured(
    question="Should we proceed?",
    options=["Yes", "No"],
    context="Critical security implications",
    strict=True
)

print(f"Choice: {response['structured']['response']}")
print(f"Reasoning: {response['structured']['reasoning']}")
print(f"Retries needed: {response['retries']}")
```

## API Reference

### `synthesize_structured()`

Main function for structured multi-agent synthesis.

**Parameters:**
- `question` (str): The question to ask all agents
- `options` (List[str]): List of options agents must choose from
- `agents` (List[Agent] | List[Dict]): Agent objects or config dicts
- `context` (Optional[str]): Additional context for agents
- `aggregation_method` (str): Aggregation method (default: "majority")
- `strict` (bool): Enforce strict validation (default: True)
- `**aggregation_params`: Method-specific parameters

**Returns:**
```python
{
    'decision': str,              # Winning option
    'decision_index': int,        # Winning option index (1-based)
    'confidence': float,          # Confidence score (0-1)
    'agent_outputs': List[Dict],  # Individual responses with reasoning
    'aggregation': Dict,          # Aggregation details
    'method': str,                # Aggregation method used
    'options': List[str],         # List of options
    'total_retries': int          # Total retries across all agents
}
```

### `Agent.generate_structured()`

Generate structured response from single agent.

**Parameters:**
- `question` (str): Question to ask
- `options` (List[str]): Options to choose from
- `context` (Optional[str]): Additional context
- `strict` (bool): Strict validation mode (default: True)

**Returns:**
```python
{
    'text': str,                    # Full response text
    'structured': StructuredResponse,  # Parsed response
    'validation': ValidationResult,    # Validation details
    'metadata': Dict,               # Provider metadata
    'retries': int                  # Number of retries used
}
```

### `Council.decide_structured()`

Make collective decision using structured responses.

**Parameters:**
- `question` (str): Question to ask all agents
- `options` (List[str]): Options to choose from
- `context` (Optional[str]): Additional context
- `strict` (bool): Strict validation (default: True)

**Returns:** Same as `synthesize_structured()`

### `ResponseValidator`

Utility class for validation and parsing.

**Methods:**

- `validate_response(text, n_options, strict=True)` → `ValidationResult`
  - Validates and parses response text
  - Returns validation result with parsed data or error

- `create_retry_prompt(original_prompt, invalid_response, error_message, n_options)` → `str`
  - Creates correction prompt for retry attempts

### `format_prompt_with_options()`

Format question with numbered options and instructions.

**Parameters:**
- `question` (str): Question to ask
- `options` (List[str]): List of options
- `context` (Optional[str]): Additional context

**Returns:** Formatted prompt string with instructions

## Validation Patterns

The system uses multiple regex patterns to detect valid responses:

### 1. JSON Pattern (Preferred)

```json
{"response": 2, "reasoning": "This is the best option because..."}
```

### 2. Colon-Separated Format

```
response: 2
reasoning: This is the best option because...
```

### 3. Lenient Mode (Non-strict)

In lenient mode, the first number in the text is extracted:

```
I choose option 2 because it's the most practical approach.
```

## Error Handling

### Validation Failures

If an agent response fails validation:

```python
try:
    result = agent.generate_structured(
        question="Choose:",
        options=["A", "B"],
        strict=True
    )
except ValueError as e:
    print(f"Validation failed: {e}")
    # Agent exhausted all retries without valid format
```

### Retry Process

```
Attempt 1: Agent provides response
  ↓ (Invalid format detected)
Retry Prompt: "Your previous response did not match the required format..."
  ↓
Attempt 2: Agent provides corrected response
  ↓ (Still invalid)
Retry Prompt: (Another correction prompt)
  ↓
Attempt 3: Agent provides response
  ↓ (Valid format)
Success: Response accepted
```

## Best Practices

### 1. Define Clear Options

```python
# ✅ Good: Clear, distinct options
options = [
    "Implement feature A",
    "Implement feature B",
    "Defer decision to next quarter"
]

# ❌ Bad: Vague or overlapping options
options = ["Maybe", "Perhaps", "Could be"]
```

### 2. Provide Sufficient Context

```python
# ✅ Good: Rich context
context = """
Current situation: Q4 planning
Budget: $100k remaining
Team capacity: 60% allocated
Risk level: Medium
Stakeholder priority: High
"""

# ❌ Bad: Insufficient context
context = "Budget low"
```

### 3. Choose Appropriate Aggregation

```python
# For simple majority voting
aggregation_method="majority"

# For inequality-averse aggregation (protects minorities)
aggregation_method="atkinson"
epsilon=1.0  # Higher = more protection

# For consensus-seeking
aggregation_method="consensus"
```

### 4. Handle Retries Appropriately

```python
# For critical decisions: more retries
agent = Agent("openai", "gpt-4", max_retries=3)

# For quick polls: fewer retries
agent = Agent("ollama", "llama3.2", max_retries=1)
```

### 5. Use Strict Mode for Production

```python
# Production: strict validation
result = council.decide_structured(
    question=question,
    options=options,
    strict=True  # Enforce exact format
)

# Development/testing: lenient mode
result = council.decide_structured(
    question=question,
    options=options,
    strict=False  # Allow informal responses
)
```

## Integration with Existing Code

The structured synthesis system is fully compatible with existing AgorAI features:

```python
from agorai import synthesize_structured, list_methods

# Use any aggregation method
methods = list_methods()
for method in methods:
    result = synthesize_structured(
        question="Choose:",
        options=["A", "B", "C"],
        agents=agents,
        aggregation_method=method['name']
    )
    print(f"{method['name']}: {result['decision']}")
```

## Performance Considerations

- **Retry overhead**: Each retry adds one LLM call per agent
- **Validation is fast**: Regex-based validation has negligible overhead
- **Parallel agent calls**: Agents are queried in parallel (where supported)
- **Binary utilities**: Structured responses use binary utilities (1.0 or 0.0), which aggregates efficiently

## Debugging Tips

### 1. Check Validation Results

```python
result = agent.generate_structured(question, options)
print(f"Valid: {result['validation'].is_valid}")
print(f"Retries: {result['retries']}")
if not result['validation'].is_valid:
    print(f"Error: {result['validation'].error_message}")
```

### 2. Inspect Raw Responses

```python
result = council.decide_structured(question, options)
for output in result['agent_outputs']:
    print(f"\nAgent: {output['agent']}")
    print(f"Raw text: {output['text']}")
    print(f"Parsed response: {output['response_number']}")
```

### 3. Test Validation Separately

```python
from agorai.synthesis.validation import ResponseValidator

test_response = '{"response": 2, "reasoning": "Test"}'
validation = ResponseValidator.validate_response(test_response, n_options=3)
print(f"Valid: {validation.is_valid}")
print(f"Parsed: {validation.parsed_response}")
```

## Examples

See `examples/structured_synthesis_demo.py` for comprehensive examples including:
- Basic structured synthesis
- Context-aware decision making
- Comparing aggregation methods
- Error handling and validation

## Testing

Run tests with:

```bash
pytest tests/test_validation.py -v
pytest tests/test_structured_synthesis.py -v
```

## FAQ

**Q: What happens if all retry attempts fail?**
A: A `ValueError` is raised with details about the validation failure.

**Q: Can I customize the prompt format?**
A: Yes, use `format_prompt_with_options()` to generate the base prompt, then modify it.

**Q: Does this work with all LLM providers?**
A: Yes, it works with any provider that returns text responses (OpenAI, Anthropic, Ollama, etc.).

**Q: Can I use this for multimodal inputs?**
A: Currently optimized for text, but multimodal support is planned.

**Q: What's the difference between strict and lenient mode?**
A: Strict requires exact format match. Lenient extracts the first valid number from any text.

## Changelog

### Version 0.2.0
- ✨ Added structured response format with validation
- ✨ Added automatic retry mechanism for malformed responses
- ✨ Added clear interfaces (TypedDicts) for all operations
- ✨ Added `synthesize_structured()` function
- ✨ Added `Agent.generate_structured()` method
- ✨ Added `Council.decide_structured()` method
- ✨ Added comprehensive validation with regex patterns
- 📚 Added extensive documentation and examples
- ✅ Added comprehensive test suite
