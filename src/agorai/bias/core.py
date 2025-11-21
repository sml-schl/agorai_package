"""Core bias mitigation functionality.

Provides the main `mitigate_bias()` function and BiasConfig.
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

from agorai.synthesis import Agent, synthesize
from agorai.bias.contexts import BiasContext, HATE_SPEECH_DETECTION


@dataclass
class BiasConfig:
    """Configuration for bias mitigation pipeline.

    Parameters
    ----------
    context : Union[str, BiasContext]
        Bias mitigation context (preset name or BiasContext object)
        Presets: "hate_speech_detection", "content_moderation", "fairness_assessment"
    providers : List[str]
        LLM providers to use (e.g., ["openai", "anthropic", "ollama"])
    models : Optional[Dict[str, str]]
        Model mapping per provider (default: uses provider defaults)
        Example: {"openai": "gpt-4", "anthropic": "claude-3-5-sonnet-20241022"}
    api_keys : Optional[Dict[str, str]]
        API keys per provider (not needed for Ollama)
        Example: {"openai": "sk-...", "anthropic": "sk-ant-..."}
    aggregation_method : str
        Aggregation method (default: "schulze_condorcet")
    cultural_perspectives : int
        Number of diverse cultural perspectives (default: 5)
    rounds : int
        Number of deliberation rounds (default: 2)
    temperature : float
        LLM sampling temperature (default: 0.7)
    system_prompts : Optional[List[str]]
        Custom system prompts for cultural perspectives
        If None, uses context defaults

    Examples
    --------
    >>> config = BiasConfig(
    ...     context="hate_speech_detection",
    ...     providers=["openai", "anthropic"],
    ...     api_keys={"openai": "sk-...", "anthropic": "sk-ant-..."},
    ...     aggregation_method="schulze_condorcet",
    ...     cultural_perspectives=5
    ... )
    """

    context: Union[str, BiasContext] = "hate_speech_detection"
    providers: List[str] = field(default_factory=lambda: ["ollama"])
    models: Optional[Dict[str, str]] = None
    api_keys: Optional[Dict[str, str]] = None
    aggregation_method: str = "schulze_condorcet"
    cultural_perspectives: int = 5
    rounds: int = 2
    temperature: float = 0.7
    system_prompts: Optional[List[str]] = None

    def __post_init__(self):
        """Validate and process configuration."""
        # Convert context string to BiasContext object
        if isinstance(self.context, str):
            from agorai.bias.contexts import get_context
            self.context = get_context(self.context)

        # Set default models per provider
        if self.models is None:
            self.models = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-sonnet-20241022",
                "ollama": "llama3.2",
                "google": "gemini-1.5-flash"
            }

        # Set default system prompts if not provided
        if self.system_prompts is None:
            self.system_prompts = self.context.cultural_prompts[:self.cultural_perspectives]


def mitigate_bias(
    input_text: str,
    input_image: Optional[str] = None,
    candidates: Optional[List[str]] = None,
    config: Optional[BiasConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """Mitigate bias through culturally-diverse multi-agent synthesis.

    Parameters
    ----------
    input_text : str
        Text input to analyze for bias
    input_image : Optional[str]
        Optional image input (file path or base64 string)
    candidates : Optional[List[str]]
        Optional candidate decisions (if None, uses context defaults)
    config : Optional[BiasConfig]
        Bias mitigation configuration (if None, uses defaults)
    **kwargs
        Additional configuration overrides

    Returns
    -------
    Dict[str, Any]
        {
            'decision': str,                    # Final bias-mitigated decision
            'confidence': float,                # Confidence score (0-1)
            'perspectives': List[Dict],         # Individual cultural perspectives
            'aggregation': Dict[str, Any],      # Aggregation details
            'fairness_metrics': Dict[str, Any], # Fairness analysis
            'context': str,                     # Context name
            'method': str                       # Aggregation method used
        }

    Examples
    --------
    >>> # Basic usage with defaults (Ollama)
    >>> result = mitigate_bias(
    ...     input_text="Is this content appropriate for all audiences?",
    ...     candidates=["appropriate", "inappropriate", "needs_review"]
    ... )
    >>> print(result['decision'])

    >>> # Advanced usage with multiple cloud providers
    >>> config = BiasConfig(
    ...     context="hate_speech_detection",
    ...     providers=["openai", "anthropic", "google"],
    ...     api_keys={
    ...         "openai": "sk-...",
    ...         "anthropic": "sk-ant-...",
    ...         "google": "..."
    ...     },
    ...     aggregation_method="atkinson",
    ...     cultural_perspectives=7,
    ...     rounds=3
    ... )
    >>> result = mitigate_bias(
    ...     input_text="Analyze this statement for bias",
    ...     config=config
    ... )

    >>> # Multimodal bias detection (with image)
    >>> result = mitigate_bias(
    ...     input_text="Does this meme contain hateful content?",
    ...     input_image="path/to/meme.jpg",
    ...     candidates=["hateful", "not_hateful", "ambiguous"]
    ... )

    Notes
    -----
    - Creates multiple agents with diverse cultural perspectives
    - Each agent analyzes the input from their cultural viewpoint
    - Decisions are aggregated using the specified fairness-aware method
    - Fairness metrics are computed to assess bias mitigation quality
    """
    # Use default config if not provided
    if config is None:
        config = BiasConfig(**kwargs)

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Use context candidates if not provided
    if candidates is None:
        candidates = config.context.candidate_labels

    # Build prompt from context template
    prompt = config.context.prompt_template.format(
        input_text=input_text,
        candidates=", ".join(candidates)
    )

    # Create agents with cultural perspectives
    agents = []
    for i, system_prompt in enumerate(config.system_prompts):
        # Cycle through providers
        provider = config.providers[i % len(config.providers)]
        model = config.models.get(provider, "default")

        # Get API key if needed
        api_key = None
        if config.api_keys:
            api_key = config.api_keys.get(provider)

        agent = Agent(
            provider=provider,
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=config.temperature,
            name=f"Cultural-Perspective-{i+1}"
        )
        agents.append(agent)

    # Prepare multimodal inputs if image provided
    multimodal_inputs = {}
    if input_image:
        # Load image as base64 if it's a file path
        if input_image.startswith('data:') or input_image.startswith('http'):
            multimodal_inputs['image'] = input_image
        else:
            import base64
            try:
                with open(input_image, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    multimodal_inputs['image'] = f"data:image/png;base64,{image_data}"
            except Exception as e:
                print(f"Warning: Could not load image: {e}")

    # Synthesize decision through multiple rounds if configured
    if config.rounds > 1:
        # Deliberative multi-round synthesis
        result = _deliberative_synthesis(
            agents=agents,
            prompt=prompt,
            candidates=candidates,
            aggregation_method=config.aggregation_method,
            rounds=config.rounds,
            multimodal_inputs=multimodal_inputs or None
        )
    else:
        # Single-round synthesis
        result = synthesize(
            prompt=prompt,
            agents=agents,
            candidates=candidates,
            aggregation_method=config.aggregation_method,
            multimodal_inputs=multimodal_inputs or None
        )

    # Compute fairness metrics
    fairness_metrics = _compute_fairness_metrics(
        agent_outputs=result['agent_outputs'],
        aggregation=result['aggregation'],
        candidates=candidates
    )

    # Format final result
    return {
        'decision': result['decision'],
        'confidence': result['confidence'],
        'perspectives': result['agent_outputs'],
        'aggregation': result['aggregation'],
        'fairness_metrics': fairness_metrics,
        'context': config.context.name,
        'method': result['method'],
        'candidates': candidates
    }


def _deliberative_synthesis(
    agents: List[Agent],
    prompt: str,
    candidates: List[str],
    aggregation_method: str,
    rounds: int,
    multimodal_inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Perform multi-round deliberative synthesis.

    Each round:
    1. Agents generate opinions
    2. Opinions are aggregated
    3. Top decision is presented to agents for next round
    """
    current_prompt = prompt
    history = []

    for round_num in range(rounds):
        # Synthesize for this round
        result = synthesize(
            prompt=current_prompt,
            agents=agents,
            candidates=candidates,
            aggregation_method=aggregation_method,
            multimodal_inputs=multimodal_inputs
        )

        history.append(result)

        # Prepare prompt for next round (if not final round)
        if round_num < rounds - 1:
            decision = result['decision']
            current_prompt = (
                f"{prompt}\n\n"
                f"[Round {round_num + 1} Result: {decision}]\n"
                f"Please reconsider your assessment given the collective decision above."
            )

    # Return final round result with history
    final_result = history[-1]
    final_result['deliberation_history'] = history

    return final_result


def _compute_fairness_metrics(
    agent_outputs: List[Dict[str, Any]],
    aggregation: Dict[str, Any],
    candidates: List[str]
) -> Dict[str, Any]:
    """Compute fairness metrics for bias mitigation.

    Metrics:
    - Diversity: How diverse are agent opinions?
    - Consensus: How much agreement exists?
    - Minority_protection: Are minority views considered?
    - Fairness_score: Overall fairness score
    """
    import statistics

    # Extract utilities from aggregation metadata
    scores = aggregation.get('scores', [])
    winner_idx = aggregation.get('winner')

    if not scores or winner_idx is None:
        return {
            'diversity': 0.0,
            'consensus': 0.0,
            'minority_protection': 0.0,
            'fairness_score': 0.0
        }

    # Diversity: coefficient of variation of scores
    if len(scores) > 1:
        mean_score = statistics.mean(scores)
        if mean_score > 1e-6:
            stdev_score = statistics.stdev(scores)
            diversity = stdev_score / mean_score
        else:
            diversity = 0.0
    else:
        diversity = 0.0

    # Consensus: winner's score relative to total
    total_score = sum(scores)
    if total_score > 0:
        consensus = scores[winner_idx] / total_score
    else:
        consensus = 0.0

    # Minority protection: ratio of min score to max score
    min_score = min(scores)
    max_score = max(scores)
    if max_score > 0:
        minority_protection = min_score / max_score
    else:
        minority_protection = 0.0

    # Overall fairness score (balanced metric)
    fairness_score = (
        0.3 * diversity +          # Reward diverse opinions
        0.3 * consensus +           # Reward clear winner
        0.4 * minority_protection   # Reward minority consideration
    )

    return {
        'diversity': round(diversity, 3),
        'consensus': round(consensus, 3),
        'minority_protection': round(minority_protection, 3),
        'fairness_score': round(fairness_score, 3)
    }
