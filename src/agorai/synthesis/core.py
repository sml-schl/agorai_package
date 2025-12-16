"""Core synthesis functionality and API.

Provides the main `synthesize()` function and Agent/Council abstractions.
"""

from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
import re

from agorai.aggregate import aggregate as agg_aggregate
from agorai.synthesis.validation import (
    ResponseValidator,
    ValidationResult,
    StructuredResponse,
    format_prompt_with_options
)


class Agent:
    """LLM agent for generating opinions.

    Parameters
    ----------
    provider : str
        LLM provider name ("openai", "anthropic", "ollama", "google")
    model : str
        Model identifier (e.g., "gpt-4", "claude-3-5-sonnet-20241022", "llama3.2")
    api_key : Optional[str]
        API key for cloud providers (not needed for Ollama)
    base_url : Optional[str]
        Custom base URL (primarily for Ollama, default: http://localhost:11434)
    system_prompt : Optional[str]
        System prompt to steer the model
    temperature : float
        Sampling temperature (0.0-2.0, default: 0.7)
    name : Optional[str]
        Human-readable agent name (default: auto-generated)

    Examples
    --------
    >>> agent1 = Agent("openai", "gpt-4", api_key="sk-...")
    >>> agent2 = Agent("ollama", "llama3.2")
    >>> agent3 = Agent("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-...")
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        name: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.name = name or f"{provider}:{model}"
        self.max_retries = max_retries

        # Import and instantiate provider
        from agorai.synthesis.providers import get_provider

        self._provider_instance = get_provider(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM.

        Parameters
        ----------
        prompt : str
            User prompt
        **kwargs
            Additional provider-specific parameters

        Returns
        -------
        Dict[str, Any]
            {
                'text': str,           # Generated response
                'scores': Dict[str, float],  # Optional candidate scores
                'metadata': Dict[str, Any]   # Provider metadata
            }
        """
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        return self._provider_instance.generate(full_prompt, **kwargs)

    def generate_structured(
        self,
        question: str,
        options: List[str],
        context: Optional[str] = None,
        strict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response with validation and retry.

        This method ensures agents return properly formatted responses with
        clear option selection and reasoning, suitable for mathematical aggregation.

        Parameters
        ----------
        question : str
            The question to ask the agent
        options : List[str]
            List of options the agent must choose from
        context : Optional[str]
            Additional context to provide to the agent
        strict : bool
            If True, enforce strict format validation. If False, attempt lenient parsing.
        **kwargs
            Additional provider-specific parameters

        Returns
        -------
        Dict[str, Any]
            {
                'text': str,                        # Full response text
                'structured': StructuredResponse,   # Parsed structured response
                'validation': ValidationResult,     # Validation details
                'metadata': Dict[str, Any],         # Provider metadata
                'retries': int                      # Number of retries used
            }

        Raises
        ------
        ValueError
            If response validation fails after all retries
        """
        if not options:
            raise ValueError("options must be a non-empty list")

        # Format prompt with numbered options and format instructions
        formatted_prompt = format_prompt_with_options(question, options, context)

        # Try to get valid response with retries
        retries_used = 0
        last_validation = None

        for attempt in range(self.max_retries + 1):
            # Generate response
            if attempt == 0:
                # First attempt with formatted prompt
                result = self.generate(formatted_prompt, **kwargs)
            else:
                # Retry with correction prompt
                retry_prompt = ResponseValidator.create_retry_prompt(
                    original_prompt=formatted_prompt,
                    invalid_response=last_validation.raw_text,
                    error_message=last_validation.error_message,
                    n_options=len(options)
                )
                result = self.generate(retry_prompt, **kwargs)
                retries_used += 1

            # Validate response
            validation = ResponseValidator.validate_response(
                text=result['text'],
                n_options=len(options),
                strict=strict
            )

            if validation.is_valid:
                # Success - return structured response
                return {
                    'text': result['text'],
                    'structured': validation.parsed_response,
                    'validation': validation,
                    'metadata': result.get('metadata', {}),
                    'retries': retries_used
                }

            last_validation = validation

        # All retries exhausted - raise error
        raise ValueError(
            f"Agent {self.name} failed to provide valid structured response after {self.max_retries} retries. "
            f"Last error: {last_validation.error_message}"
        )

    def __repr__(self):
        return f"Agent({self.provider}:{self.model}, name={self.name})"


class Council:
    """Multi-agent council for democratic opinion synthesis.

    Parameters
    ----------
    agents : List[Agent]
        List of agents to participate in the council
    aggregation_method : str
        Aggregation method from agorai.aggregate (default: "majority")
    **aggregation_params
        Additional parameters for the aggregation method

    Examples
    --------
    >>> agents = [Agent("openai", "gpt-4"), Agent("anthropic", "claude-3-5-sonnet-20241022")]
    >>> council = Council(agents, aggregation_method="atkinson", epsilon=1.0)
    >>> result = council.decide("Should we approve this?", candidates=["approve", "reject"])
    """

    def __init__(
        self,
        agents: List[Agent],
        aggregation_method: str = "majority",
        **aggregation_params
    ):
        if not agents:
            raise ValueError("Council must have at least one agent")

        self.agents = agents
        self.aggregation_method = aggregation_method
        self.aggregation_params = aggregation_params

    def decide(
        self,
        prompt: str,
        candidates: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a collective decision through democratic aggregation.

        Parameters
        ----------
        prompt : str
            Prompt to send to all agents
        candidates : Optional[List[str]]
            List of candidate options (if None, extracted from agent outputs)
        **kwargs
            Additional generation parameters

        Returns
        -------
        Dict[str, Any]
            {
                'decision': str,                    # Winning candidate/decision
                'confidence': float,                # Confidence score
                'agent_outputs': List[Dict],        # Individual agent responses
                'aggregation': Dict[str, Any],      # Aggregation details
                'method': str                       # Aggregation method used
            }
        """
        # Collect agent outputs
        agent_outputs = []
        for agent in self.agents:
            output = agent.generate(prompt, **kwargs)
            agent_outputs.append({
                'agent': agent.name,
                'text': output['text'],
                'scores': output.get('scores', {}),
                'metadata': output.get('metadata', {})
            })

        # Extract candidates if not provided
        if candidates is None:
            candidates = self._extract_candidates(agent_outputs)

        # Convert agent outputs to utilities
        utilities = self._outputs_to_utilities(agent_outputs, candidates)

        # Aggregate using agorai.aggregate
        agg_result = agg_aggregate(
            utilities=utilities,
            method=self.aggregation_method,
            **self.aggregation_params
        )

        winner_idx = agg_result['winner']
        decision = candidates[winner_idx] if winner_idx is not None else None

        # Calculate confidence (normalized winning score)
        scores = agg_result.get('scores', [])
        if scores:
            max_score = max(scores) if scores else 0
            total_score = sum(scores) if scores else 1
            confidence = max_score / total_score if total_score > 0 else 0.0
        else:
            confidence = 0.0

        return {
            'decision': decision,
            'confidence': confidence,
            'agent_outputs': agent_outputs,
            'aggregation': agg_result,
            'method': self.aggregation_method,
            'candidates': candidates
        }

    def decide_structured(
        self,
        question: str,
        options: List[str],
        context: Optional[str] = None,
        strict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a collective decision using structured responses.

        This method ensures all agents provide properly formatted responses with
        clear option selection, enabling robust mathematical aggregation.

        Parameters
        ----------
        question : str
            The question to ask all agents
        options : List[str]
            List of options agents must choose from
        context : Optional[str]
            Additional context to provide to agents
        strict : bool
            If True, enforce strict format validation. If False, attempt lenient parsing.
        **kwargs
            Additional generation parameters

        Returns
        -------
        Dict[str, Any]
            {
                'decision': str,                    # Winning option
                'decision_index': int,              # Winning option index (1-based)
                'confidence': float,                # Confidence score
                'agent_outputs': List[Dict],        # Individual agent responses
                'aggregation': Dict[str, Any],      # Aggregation details
                'method': str,                      # Aggregation method used
                'options': List[str],               # List of options
                'total_retries': int                # Total retries across all agents
            }

        Raises
        ------
        ValueError
            If any agent fails to provide valid structured response after retries
        """
        if not options:
            raise ValueError("options must be a non-empty list")

        # Collect structured agent outputs
        agent_outputs = []
        total_retries = 0

        for agent in self.agents:
            try:
                output = agent.generate_structured(
                    question=question,
                    options=options,
                    context=context,
                    strict=strict,
                    **kwargs
                )
                agent_outputs.append({
                    'agent': agent.name,
                    'text': output['text'],
                    'structured': output['structured'],
                    'response_number': output['structured']['response'],
                    'reasoning': output['structured']['reasoning'],
                    'metadata': output['metadata'],
                    'retries': output['retries']
                })
                total_retries += output['retries']
            except ValueError as e:
                raise ValueError(
                    f"Agent {agent.name} failed to provide valid structured response: {str(e)}"
                ) from e

        # Build utility matrix from structured responses
        # Each agent gets utility 1.0 for chosen option, 0.0 for others
        utilities = []
        for output in agent_outputs:
            chosen_idx = output['response_number'] - 1  # Convert to 0-based index
            u = [0.0] * len(options)
            u[chosen_idx] = 1.0
            utilities.append(u)

        # Aggregate using agorai.aggregate
        agg_result = agg_aggregate(
            utilities=utilities,
            method=self.aggregation_method,
            normalize=False,  # Already normalized (binary utilities)
            **self.aggregation_params
        )

        winner_idx = agg_result['winner']
        decision = options[winner_idx] if winner_idx is not None else None

        # Calculate confidence (normalized winning score)
        scores = agg_result.get('scores', [])
        if scores:
            max_score = max(scores) if scores else 0
            total_score = sum(scores) if scores else 1
            confidence = max_score / total_score if total_score > 0 else 0.0
        else:
            confidence = 0.0

        return {
            'decision': decision,
            'decision_index': winner_idx + 1 if winner_idx is not None else None,  # 1-based
            'confidence': confidence,
            'agent_outputs': agent_outputs,
            'aggregation': agg_result,
            'method': self.aggregation_method,
            'options': options,
            'total_retries': total_retries
        }

    def _extract_candidates(self, agent_outputs: List[Dict[str, Any]]) -> List[str]:
        """Extract unique candidates from agent text outputs."""
        candidates = set()

        for output in agent_outputs:
            text = output['text']
            # Extract candidate mentions from text (simple heuristic)
            # This is a fallback - better to provide explicit candidates
            words = re.findall(r'\b\w+\b', text.lower())
            candidates.update(words[:10])  # Limit to first 10 words

        return sorted(list(candidates))[:10]  # Max 10 candidates

    def _outputs_to_utilities(
        self,
        agent_outputs: List[Dict[str, Any]],
        candidates: List[str]
    ) -> List[List[float]]:
        """Convert agent outputs to utility matrix.

        Strategies:
        1. If agent provides explicit scores dict, use those
        2. Otherwise, use token overlap similarity between text and candidates
        """
        utilities = []

        for output in agent_outputs:
            text = output['text'].lower()
            scores_dict = output.get('scores', {})

            if scores_dict:
                # Use explicit scores
                u = [scores_dict.get(c, 0.0) for c in candidates]
            else:
                # Use token overlap
                text_tokens = set(re.findall(r'\b\w+\b', text))
                u = []
                for candidate in candidates:
                    cand_tokens = set(re.findall(r'\b\w+\b', candidate.lower()))
                    overlap = len(text_tokens & cand_tokens)
                    u.append(float(overlap))

            utilities.append(u)

        return utilities


def synthesize(
    prompt: str,
    agents: Union[List[Agent], List[Dict[str, Any]]],
    candidates: Optional[List[str]] = None,
    aggregation_method: str = "majority",
    multimodal_inputs: Optional[Dict[str, Any]] = None,
    **aggregation_params
) -> Dict[str, Any]:
    """Synthesize opinions from multiple LLM agents into a collective decision.

    Parameters
    ----------
    prompt : str
        Prompt to send to all agents
    agents : List[Agent] or List[Dict]
        List of Agent objects or agent configuration dicts
        Config dict format: {'provider': str, 'model': str, 'api_key': str, ...}
    candidates : Optional[List[str]]
        List of candidate options to choose from
        If None, candidates are extracted from agent responses
    aggregation_method : str
        Aggregation method from agorai.aggregate (default: "majority")
        Options: "majority", "borda", "atkinson", "maximin", etc.
    multimodal_inputs : Optional[Dict[str, Any]]
        Optional multimodal inputs (e.g., {'image': 'base64...', 'audio': '...'})
    **aggregation_params
        Additional parameters for the aggregation method
        Examples: epsilon for atkinson, threshold for approval_voting, etc.

    Returns
    -------
    Dict[str, Any]
        {
            'decision': str,                    # Winning candidate/decision
            'confidence': float,                # Confidence score (0-1)
            'agent_outputs': List[Dict],        # Individual agent responses
            'aggregation': Dict[str, Any],      # Aggregation details
            'method': str,                      # Aggregation method used
            'candidates': List[str]             # List of candidates considered
        }

    Examples
    --------
    >>> # Simple synthesis with explicit candidates
    >>> agents = [
    ...     Agent("openai", "gpt-4", api_key="sk-..."),
    ...     Agent("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
    ...     Agent("ollama", "llama3.2")
    ... ]
    >>> result = synthesize(
    ...     prompt="Should we approve this marketing campaign?",
    ...     agents=agents,
    ...     candidates=["approve", "reject", "revise"],
    ...     aggregation_method="majority"
    ... )
    >>> print(result['decision'])  # 'approve'

    >>> # Using agent config dicts
    >>> agent_configs = [
    ...     {'provider': 'openai', 'model': 'gpt-4', 'api_key': 'sk-...'},
    ...     {'provider': 'ollama', 'model': 'llama3.2'}
    ... ]
    >>> result = synthesize(
    ...     prompt="Classify sentiment: This product is amazing!",
    ...     agents=agent_configs,
    ...     candidates=["positive", "negative", "neutral"],
    ...     aggregation_method="weighted_plurality"
    ... )

    Notes
    -----
    - Agent outputs are converted to utilities using either explicit scores
      or token overlap similarity with candidates
    - All aggregation methods from agorai.aggregate are supported
    - Multimodal inputs are passed to agents that support them
    """
    # Convert config dicts to Agent objects if needed
    agent_objects = []
    for agent in agents:
        if isinstance(agent, dict):
            agent_objects.append(Agent(**agent))
        else:
            agent_objects.append(agent)

    # Create council and make decision
    council = Council(
        agents=agent_objects,
        aggregation_method=aggregation_method,
        **aggregation_params
    )

    # Add multimodal inputs to prompt if provided
    kwargs = {}
    if multimodal_inputs:
        kwargs['multimodal_inputs'] = multimodal_inputs

    return council.decide(prompt=prompt, candidates=candidates, **kwargs)


def synthesize_structured(
    question: str,
    options: List[str],
    agents: Union[List[Agent], List[Dict[str, Any]]],
    context: Optional[str] = None,
    aggregation_method: str = "majority",
    strict: bool = True,
    **aggregation_params
) -> Dict[str, Any]:
    """Synthesize opinions using structured responses with validation and retry.

    This function ensures all agents provide properly formatted responses with
    clear option selection and reasoning, enabling robust mathematical aggregation.

    Parameters
    ----------
    question : str
        The question to ask all agents
    options : List[str]
        List of options agents must choose from
    agents : List[Agent] or List[Dict]
        List of Agent objects or agent configuration dicts
        Config dict format: {'provider': str, 'model': str, 'api_key': str, ...}
    context : Optional[str]
        Additional context to provide to agents
    aggregation_method : str
        Aggregation method from agorai.aggregate (default: "majority")
        Options: "majority", "borda", "atkinson", "maximin", etc.
    strict : bool
        If True, enforce strict format validation. If False, attempt lenient parsing.
    **aggregation_params
        Additional parameters for the aggregation method
        Examples: epsilon for atkinson, threshold for approval_voting, etc.

    Returns
    -------
    Dict[str, Any]
        {
            'decision': str,                    # Winning option
            'decision_index': int,              # Winning option index (1-based)
            'confidence': float,                # Confidence score (0-1)
            'agent_outputs': List[Dict],        # Individual agent responses with reasoning
            'aggregation': Dict[str, Any],      # Aggregation details
            'method': str,                      # Aggregation method used
            'options': List[str],               # List of options
            'total_retries': int                # Total retries across all agents
        }

    Raises
    ------
    ValueError
        If options is empty or if any agent fails to provide valid structured response

    Examples
    --------
    >>> # Structured synthesis with explicit options
    >>> agents = [
    ...     Agent("openai", "gpt-4", api_key="sk-..."),
    ...     Agent("anthropic", "claude-3-5-sonnet-20241022", api_key="sk-ant-..."),
    ...     Agent("ollama", "llama3.2")
    ... ]
    >>> result = synthesize_structured(
    ...     question="Should we approve this marketing campaign?",
    ...     options=["approve", "reject", "revise"],
    ...     agents=agents,
    ...     aggregation_method="majority"
    ... )
    >>> print(result['decision'])  # 'approve'
    >>> print(result['agent_outputs'][0]['reasoning'])  # Agent's explanation

    >>> # With additional context
    >>> result = synthesize_structured(
    ...     question="Which approach should we take?",
    ...     options=["approach A", "approach B", "approach C"],
    ...     agents=agent_configs,
    ...     context="Budget: $10k, Timeline: 3 months",
    ...     aggregation_method="atkinson",
    ...     epsilon=1.0
    ... )

    Notes
    -----
    - Each agent must return a structured response: {"response": <number>, "reasoning": <text>}
    - Invalid responses trigger automatic retry with format correction prompt
    - All responses are validated with regex before aggregation
    - Aggregation uses binary utilities: 1.0 for chosen option, 0.0 for others
    """
    if not options:
        raise ValueError("options must be a non-empty list")

    # Convert config dicts to Agent objects if needed
    agent_objects = []
    for agent in agents:
        if isinstance(agent, dict):
            agent_objects.append(Agent(**agent))
        else:
            agent_objects.append(agent)

    # Create council and make structured decision
    council = Council(
        agents=agent_objects,
        aggregation_method=aggregation_method,
        **aggregation_params
    )

    return council.decide_structured(
        question=question,
        options=options,
        context=context,
        strict=strict
    )
