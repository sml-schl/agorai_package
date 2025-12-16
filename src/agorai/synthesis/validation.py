"""Validation and response format enforcement for structured agent responses.

This module provides structured response formats, validation, and retry mechanisms
to ensure agents return clean, parseable decisions suitable for mathematical aggregation.
"""

from typing import Dict, Any, Optional, List, TypedDict
from dataclasses import dataclass
import re
import json


class StructuredResponse(TypedDict):
    """Structured response format for agent decisions.

    Attributes
    ----------
    response : int
        The chosen option number (1-based indexing matching the options)
    reasoning : str
        Explanation for why this option was chosen
    """
    response: int
    reasoning: str


@dataclass
class ValidationResult:
    """Result of response validation.

    Attributes
    ----------
    is_valid : bool
        Whether the response passes validation
    parsed_response : Optional[StructuredResponse]
        Parsed structured response if valid
    error_message : Optional[str]
        Error message if validation failed
    raw_text : str
        Original raw response text
    """
    is_valid: bool
    parsed_response: Optional[StructuredResponse]
    error_message: Optional[str]
    raw_text: str


class ResponseValidator:
    """Validates and parses agent responses into structured format."""

    # Regex patterns for extracting structured responses
    JSON_PATTERN = re.compile(
        r'\{[^{}]*"response"\s*:\s*(\d+)[^{}]*"reasoning"\s*:\s*"([^"]*)"[^{}]*\}',
        re.DOTALL | re.IGNORECASE
    )

    # Alternative pattern: response: 1, reasoning: ...
    COLON_PATTERN = re.compile(
        r'response\s*:\s*(\d+).*?reasoning\s*:\s*(.+?)(?:\n\n|$)',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern to extract just numbers (fallback)
    NUMBER_PATTERN = re.compile(r'\b(\d+)\b')

    @staticmethod
    def validate_response(
        text: str,
        n_options: int,
        strict: bool = True
    ) -> ValidationResult:
        """Validate and parse agent response text.

        Parameters
        ----------
        text : str
            Raw response text from agent
        n_options : int
            Number of valid options (for range checking)
        strict : bool
            If True, require exact format match. If False, attempt lenient parsing.

        Returns
        -------
        ValidationResult
            Validation result with parsed response or error message
        """
        # Try JSON pattern first
        json_match = ResponseValidator.JSON_PATTERN.search(text)
        if json_match:
            response_num = int(json_match.group(1))
            reasoning = json_match.group(2).strip()

            if 1 <= response_num <= n_options:
                return ValidationResult(
                    is_valid=True,
                    parsed_response=StructuredResponse(
                        response=response_num,
                        reasoning=reasoning
                    ),
                    error_message=None,
                    raw_text=text
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    parsed_response=None,
                    error_message=f"Response number {response_num} out of range [1, {n_options}]",
                    raw_text=text
                )

        # Try colon pattern
        colon_match = ResponseValidator.COLON_PATTERN.search(text)
        if colon_match:
            response_num = int(colon_match.group(1))
            reasoning = colon_match.group(2).strip()

            if 1 <= response_num <= n_options:
                return ValidationResult(
                    is_valid=True,
                    parsed_response=StructuredResponse(
                        response=response_num,
                        reasoning=reasoning
                    ),
                    error_message=None,
                    raw_text=text
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    parsed_response=None,
                    error_message=f"Response number {response_num} out of range [1, {n_options}]",
                    raw_text=text
                )

        # Try to parse as JSON directly
        try:
            # Look for JSON-like structure
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)

                if 'response' in parsed and 'reasoning' in parsed:
                    response_num = int(parsed['response'])
                    reasoning = str(parsed['reasoning'])

                    if 1 <= response_num <= n_options:
                        return ValidationResult(
                            is_valid=True,
                            parsed_response=StructuredResponse(
                                response=response_num,
                                reasoning=reasoning
                            ),
                            error_message=None,
                            raw_text=text
                        )
                    else:
                        return ValidationResult(
                            is_valid=False,
                            parsed_response=None,
                            error_message=f"Response number {response_num} out of range [1, {n_options}]",
                            raw_text=text
                        )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Lenient mode: try to extract first number
        if not strict:
            numbers = ResponseValidator.NUMBER_PATTERN.findall(text)
            if numbers:
                response_num = int(numbers[0])
                if 1 <= response_num <= n_options:
                    return ValidationResult(
                        is_valid=True,
                        parsed_response=StructuredResponse(
                            response=response_num,
                            reasoning=text.strip()  # Use full text as reasoning
                        ),
                        error_message=None,
                        raw_text=text
                    )

        # Validation failed
        return ValidationResult(
            is_valid=False,
            parsed_response=None,
            error_message=(
                "Response does not match required format. "
                "Expected format: {\"response\": <number>, \"reasoning\": \"<explanation>\"}"
            ),
            raw_text=text
        )

    @staticmethod
    def create_retry_prompt(
        original_prompt: str,
        invalid_response: str,
        error_message: str,
        n_options: int
    ) -> str:
        """Create a retry prompt for malformed responses.

        Parameters
        ----------
        original_prompt : str
            Original prompt that was sent
        invalid_response : str
            The invalid response that was received
        error_message : str
            Validation error message
        n_options : int
            Number of valid options

        Returns
        -------
        str
            Retry prompt with format instructions
        """
        return f"""Your previous response did not match the required format.

ERROR: {error_message}

REQUIRED FORMAT:
You MUST respond using EXACTLY this JSON format:
{{
    "response": <number between 1 and {n_options}>,
    "reasoning": "<your explanation>"
}}

ORIGINAL PROMPT:
{original_prompt}

YOUR PREVIOUS INVALID RESPONSE:
{invalid_response}

Please provide your response again in the correct format. Only provide the JSON object, nothing else."""


def format_prompt_with_options(
    question: str,
    options: List[str],
    context: Optional[str] = None
) -> str:
    """Format a prompt with numbered options and structured response instructions.

    Parameters
    ----------
    question : str
        The question to ask
    options : List[str]
        List of options to choose from
    context : Optional[str]
        Additional context to include in the prompt

    Returns
    -------
    str
        Formatted prompt with options and format instructions
    """
    formatted_options = []
    for i, option in enumerate(options, start=1):
        formatted_options.append(f"{i}. {option}")

    options_text = "\n".join(formatted_options)

    prompt = f"""{question}

OPTIONS:
{options_text}
"""

    if context:
        prompt += f"""
CONTEXT:
{context}
"""

    prompt += f"""
INSTRUCTIONS:
You must choose exactly ONE option from the list above and respond in the following structured format:

{{
    "response": <option number>,
    "reasoning": "<brief explanation for your choice>"
}}

IMPORTANT:
- "response" must be a single number from 1 to {len(options)}
- "reasoning" should briefly explain why you chose this option
- Only provide the JSON object, nothing else
"""

    return prompt
