"""Automatic council creation using LLM-based perspective identification.

This module provides functionality to automatically generate a council of agents
based on a question, using an LLM to identify relevant perspectives.
"""

from typing import List, Dict, Optional, Any
import re
import json
from agorai.synthesis.core import Agent, Council
from agorai.logging_config import get_logger

logger = get_logger(__name__)


# System prompt for the meta-LLM that designs the council
COUNCIL_DESIGN_PROMPT = """You are a council design expert. Your task is to identify diverse, relevant perspectives for a multi-agent discussion system.

Given a question or decision scenario, identify 3-7 distinct perspectives that should be represented. Each perspective should:
1. Be genuinely different from others
2. Be relevant to the question
3. Represent a stakeholder or viewpoint that adds value
4. Have clear expertise or interest in the topic

For each perspective, provide:
- A concise name (2-4 words, e.g., "Worker Representative", "Legal Expert")
- A system prompt that defines their role, expertise, and how they should approach the question

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "perspectives": [
    {
      "name": "Perspective Name",
      "system_prompt": "You are a [role]. Your expertise is in [domain]. When evaluating questions, you focus on [key concerns]."
    }
  ],
  "reasoning": "Brief explanation of why these perspectives were chosen"
}

Ensure:
- Generate between 3 and 7 perspectives
- Names are concise and descriptive
- System prompts are 2-4 sentences
- JSON is valid and parseable
- No markdown formatting, just pure JSON"""


def parse_council_response(response: str) -> Dict[str, Any]:
    """Parse and validate council design response.

    Parameters
    ----------
    response : str
        Raw response from meta-LLM

    Returns
    -------
    Dict[str, Any]
        Parsed response with perspectives and reasoning

    Raises
    ------
    ValueError
        If response format is invalid
    """
    # Try to extract JSON from response
    # Handle cases where LLM might wrap JSON in markdown or add text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        raise ValueError("No JSON found in response")

    json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate structure
    if 'perspectives' not in data:
        raise ValueError("Response missing 'perspectives' field")

    perspectives = data['perspectives']
    if not isinstance(perspectives, list):
        raise ValueError("'perspectives' must be a list")

    if not (3 <= len(perspectives) <= 7):
        raise ValueError(f"Expected 3-7 perspectives, got {len(perspectives)}")

    # Validate each perspective
    for i, persp in enumerate(perspectives):
        if not isinstance(persp, dict):
            raise ValueError(f"Perspective {i} is not a dict")

        if 'name' not in persp or 'system_prompt' not in persp:
            raise ValueError(f"Perspective {i} missing 'name' or 'system_prompt'")

        if not isinstance(persp['name'], str) or not persp['name'].strip():
            raise ValueError(f"Perspective {i} has invalid name")

        if not isinstance(persp['system_prompt'], str) or not persp['system_prompt'].strip():
            raise ValueError(f"Perspective {i} has invalid system_prompt")

    logger.info(f"Parsed {len(perspectives)} perspectives from council design response")
    return data


def create_automatic_council(
    question: str,
    context: Optional[str] = None,
    automatic_llm_provider: str = "anthropic",
    automatic_llm_model: str = "claude-3-5-sonnet-20241022",
    automatic_llm_api_key: Optional[str] = None,
    automatic_llm_base_url: Optional[str] = None,
    agent_provider: str = "anthropic",
    agent_model: str = "claude-3-5-sonnet-20241022",
    agent_api_key: Optional[str] = None,
    agent_base_url: Optional[str] = None,
    agent_temperature: float = 0.7,
    aggregation_method: str = "majority",
    **aggregation_params
) -> Dict[str, Any]:
    """Automatically create a council based on a question.

    This function uses a meta-LLM to identify relevant perspectives for a question,
    then creates a council with agents representing those perspectives.

    Parameters
    ----------
    question : str
        The question or decision scenario to create a council for
    context : Optional[str]
        Additional context about the question
    automatic_llm_provider : str
        Provider for the meta-LLM that designs the council (default: "anthropic")
    automatic_llm_model : str
        Model for the meta-LLM (default: "claude-3-5-sonnet-20241022")
    automatic_llm_api_key : Optional[str]
        API key for the meta-LLM provider
    automatic_llm_base_url : Optional[str]
        Base URL for the meta-LLM provider (for Ollama)
    agent_provider : str
        Provider for council agents (default: "anthropic")
    agent_model : str
        Model for council agents (default: "claude-3-5-sonnet-20241022")
    agent_api_key : Optional[str]
        API key for agent provider
    agent_base_url : Optional[str]
        Base URL for agent provider (for Ollama)
    agent_temperature : float
        Temperature for council agents (default: 0.7)
    aggregation_method : str
        Aggregation method for the council (default: "majority")
    **aggregation_params
        Additional parameters for the aggregation method

    Returns
    -------
    Dict[str, Any]
        {
            'council': Council,                      # The created council
            'perspectives': List[Dict[str, str]],    # List of perspectives with names and prompts
            'reasoning': str,                        # Explanation of perspective choices
            'meta_response': str                     # Raw response from meta-LLM
        }

    Raises
    ------
    ValueError
        If the meta-LLM fails to generate valid perspectives
    TimeoutError
        If LLM generation times out

    Examples
    --------
    >>> result = create_automatic_council(
    ...     question="Should we terminate this employee for misconduct?",
    ...     automatic_llm_api_key="sk-ant-...",
    ...     agent_api_key="sk-ant-..."
    ... )
    >>> council = result['council']
    >>> print([p['name'] for p in result['perspectives']])
    ['HR Manager', 'Employee Representative', 'Legal Counsel', 'Ethics Officer']

    >>> # Use the council
    >>> decision = council.decide_structured(
    ...     question="Should we terminate this employee?",
    ...     options=["Terminate", "Warning", "No Action"]
    ... )
    """
    logger.info(f"Creating automatic council for question: {question[:100]}...")

    # Build prompt for meta-LLM
    full_prompt = f"Question/Scenario: {question}"
    if context:
        full_prompt += f"\n\nContext: {context}"

    # Create meta-LLM agent
    meta_agent = Agent(
        provider=automatic_llm_provider,
        model=automatic_llm_model,
        api_key=automatic_llm_api_key,
        base_url=automatic_llm_base_url,
        system_prompt=COUNCIL_DESIGN_PROMPT,
        temperature=0.3,  # Lower temperature for more consistent output
        name="CouncilDesigner"
    )

    logger.debug(f"Meta-LLM agent created: {meta_agent.name}")

    # Generate council design
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting council design generation (attempt {attempt + 1}/{max_retries})")
            response = meta_agent.generate(full_prompt)
            raw_response = response['text']

            logger.debug(f"Raw meta-LLM response: {raw_response[:200]}...")

            # Parse and validate response
            parsed = parse_council_response(raw_response)

            perspectives = parsed['perspectives']
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            logger.info(f"Successfully designed council with {len(perspectives)} perspectives")

            # Create council agents
            agents = []
            for persp in perspectives:
                agent = Agent(
                    provider=agent_provider,
                    model=agent_model,
                    api_key=agent_api_key,
                    base_url=agent_base_url,
                    system_prompt=persp['system_prompt'],
                    temperature=agent_temperature,
                    name=persp['name']
                )
                agents.append(agent)
                logger.debug(f"Created agent: {agent.name}")

            # Create council
            council = Council(
                agents=agents,
                aggregation_method=aggregation_method,
                **aggregation_params
            )

            logger.info(f"Council created successfully with {len(agents)} agents")

            return {
                'council': council,
                'perspectives': perspectives,
                'reasoning': reasoning,
                'meta_response': raw_response
            }

        except ValueError as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying council design...")
                continue
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise ValueError(
                    f"Failed to create automatic council after {max_retries} attempts. "
                    f"Last error: {e}"
                )

        except Exception as e:
            logger.error(f"Unexpected error during council creation: {e}")
            raise


def create_automatic_council_simple(
    question: str,
    api_key: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    aggregation_method: str = "majority",
    **aggregation_params
) -> Council:
    """Simplified automatic council creation using same provider for all agents.

    This is a convenience function that uses the same provider/model for both
    the meta-LLM and the council agents.

    Parameters
    ----------
    question : str
        The question or decision scenario
    api_key : str
        API key for the LLM provider
    provider : str
        LLM provider (default: "anthropic")
    model : str
        LLM model (default: "claude-3-5-sonnet-20241022")
    aggregation_method : str
        Aggregation method (default: "majority")
    **aggregation_params
        Additional aggregation parameters

    Returns
    -------
    Council
        The created council ready to use

    Examples
    --------
    >>> council = create_automatic_council_simple(
    ...     question="Should we approve this budget?",
    ...     api_key="sk-ant-..."
    ... )
    >>> decision = council.decide_structured(
    ...     question="Should we approve this budget?",
    ...     options=["Approve", "Reject", "Request Revision"]
    ... )
    """
    result = create_automatic_council(
        question=question,
        automatic_llm_provider=provider,
        automatic_llm_model=model,
        automatic_llm_api_key=api_key,
        agent_provider=provider,
        agent_model=model,
        agent_api_key=api_key,
        aggregation_method=aggregation_method,
        **aggregation_params
    )

    return result['council']
