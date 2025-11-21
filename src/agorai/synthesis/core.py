"""Core synthesis functionality and API.

Provides the main `synthesize()` function and Agent/Council abstractions.
"""

from typing import List, Dict, Optional, Any, Union
from abc import ABC, abstractmethod
import re

from agorai.aggregate import aggregate as agg_aggregate


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
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.name = name or f"{provider}:{model}"

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
