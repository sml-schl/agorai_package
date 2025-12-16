"""Tests for structured response validation."""

import pytest
from agorai.synthesis.validation import (
    ResponseValidator,
    ValidationResult,
    StructuredResponse,
    format_prompt_with_options
)


class TestResponseValidator:
    """Test suite for ResponseValidator."""

    def test_valid_json_format(self):
        """Test validation of properly formatted JSON response."""
        text = '{"response": 2, "reasoning": "This is the best option because..."}'
        result = ResponseValidator.validate_response(text, n_options=3)

        assert result.is_valid
        assert result.parsed_response is not None
        assert result.parsed_response['response'] == 2
        assert "best option" in result.parsed_response['reasoning']
        assert result.error_message is None

    def test_valid_json_with_extra_text(self):
        """Test validation of JSON response with surrounding text."""
        text = '''
        Here is my analysis:
        {"response": 1, "reasoning": "Clear choice"}
        That's my final answer.
        '''
        result = ResponseValidator.validate_response(text, n_options=3)

        assert result.is_valid
        assert result.parsed_response['response'] == 1
        assert result.parsed_response['reasoning'] == "Clear choice"

    def test_valid_colon_format(self):
        """Test validation of colon-separated format."""
        text = """
        response: 3
        reasoning: This option provides the best balance between cost and quality.
        """
        result = ResponseValidator.validate_response(text, n_options=3)

        assert result.is_valid
        assert result.parsed_response['response'] == 3
        assert "balance" in result.parsed_response['reasoning']

    def test_response_out_of_range(self):
        """Test validation failure when response number is out of range."""
        text = '{"response": 5, "reasoning": "Invalid choice"}'
        result = ResponseValidator.validate_response(text, n_options=3)

        assert not result.is_valid
        assert result.parsed_response is None
        assert "out of range" in result.error_message.lower()

    def test_invalid_format(self):
        """Test validation failure with completely invalid format."""
        text = "I think option B is the best choice."
        result = ResponseValidator.validate_response(text, n_options=3, strict=True)

        assert not result.is_valid
        assert result.parsed_response is None
        assert "does not match required format" in result.error_message

    def test_lenient_parsing(self):
        """Test lenient parsing extracts first number."""
        text = "I choose 2 because it's the best."
        result = ResponseValidator.validate_response(text, n_options=3, strict=False)

        assert result.is_valid
        assert result.parsed_response['response'] == 2
        assert result.parsed_response['reasoning'] == text.strip()

    def test_lenient_parsing_out_of_range(self):
        """Test lenient parsing rejects out-of-range numbers."""
        text = "I choose 5 because it's the best."
        result = ResponseValidator.validate_response(text, n_options=3, strict=False)

        assert not result.is_valid

    def test_multiline_reasoning(self):
        """Test validation with multiline reasoning."""
        text = '''{
            "response": 1,
            "reasoning": "This option is best because:
            1. It's cost-effective
            2. It's scalable
            3. It's proven"
        }'''
        result = ResponseValidator.validate_response(text, n_options=2)

        assert result.is_valid
        assert result.parsed_response['response'] == 1
        assert "cost-effective" in result.parsed_response['reasoning']

    def test_edge_case_single_option(self):
        """Test validation with single option."""
        text = '{"response": 1, "reasoning": "Only choice"}'
        result = ResponseValidator.validate_response(text, n_options=1)

        assert result.is_valid
        assert result.parsed_response['response'] == 1

    def test_edge_case_many_options(self):
        """Test validation with many options."""
        text = '{"response": 15, "reasoning": "Option 15"}'
        result = ResponseValidator.validate_response(text, n_options=20)

        assert result.is_valid
        assert result.parsed_response['response'] == 15


class TestRetryPrompt:
    """Test suite for retry prompt generation."""

    def test_create_retry_prompt(self):
        """Test retry prompt creation."""
        original_prompt = "Choose the best option"
        invalid_response = "I prefer option B"
        error_message = "Invalid format"
        n_options = 3

        retry_prompt = ResponseValidator.create_retry_prompt(
            original_prompt=original_prompt,
            invalid_response=invalid_response,
            error_message=error_message,
            n_options=n_options
        )

        assert "previous response" in retry_prompt.lower()
        assert error_message in retry_prompt
        assert original_prompt in retry_prompt
        assert invalid_response in retry_prompt
        assert "required format" in retry_prompt.lower()
        assert str(n_options) in retry_prompt

    def test_retry_prompt_includes_format_spec(self):
        """Test retry prompt includes format specification."""
        retry_prompt = ResponseValidator.create_retry_prompt(
            original_prompt="Test",
            invalid_response="Invalid",
            error_message="Error",
            n_options=5
        )

        assert "response" in retry_prompt.lower()
        assert "reasoning" in retry_prompt.lower()
        assert "json" in retry_prompt.lower()


class TestFormatPromptWithOptions:
    """Test suite for prompt formatting."""

    def test_basic_formatting(self):
        """Test basic prompt formatting with options."""
        question = "Which option is best?"
        options = ["Option A", "Option B", "Option C"]

        prompt = format_prompt_with_options(question, options)

        assert question in prompt
        assert "1. Option A" in prompt
        assert "2. Option B" in prompt
        assert "3. Option C" in prompt
        assert "response" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_formatting_with_context(self):
        """Test prompt formatting with additional context."""
        question = "Which option is best?"
        options = ["Option A", "Option B"]
        context = "Budget: $1000, Timeline: 1 month"

        prompt = format_prompt_with_options(question, options, context)

        assert question in prompt
        assert context in prompt
        assert "CONTEXT" in prompt

    def test_formatting_instructions_present(self):
        """Test that formatting instructions are included."""
        prompt = format_prompt_with_options(
            question="Test?",
            options=["A", "B", "C"]
        )

        assert "INSTRUCTIONS" in prompt
        assert "json" in prompt.lower()
        assert "must choose exactly ONE option" in prompt.lower()
        assert "1 to 3" in prompt

    def test_single_option(self):
        """Test formatting with single option."""
        prompt = format_prompt_with_options(
            question="Is this correct?",
            options=["Yes"]
        )

        assert "1. Yes" in prompt
        assert "1 to 1" in prompt

    def test_many_options(self):
        """Test formatting with many options."""
        options = [f"Option {i}" for i in range(1, 11)]
        prompt = format_prompt_with_options(
            question="Choose one:",
            options=options
        )

        assert "1. Option 1" in prompt
        assert "10. Option 10" in prompt
        assert "1 to 10" in prompt


class TestIntegration:
    """Integration tests for validation workflow."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow from prompt to parsed response."""
        # 1. Format prompt
        question = "Should we proceed?"
        options = ["Yes", "No", "Maybe"]
        prompt = format_prompt_with_options(question, options)

        # Simulate agent response
        agent_response = '{"response": 2, "reasoning": "Need more information"}'

        # 2. Validate response
        validation = ResponseValidator.validate_response(
            text=agent_response,
            n_options=len(options)
        )

        # 3. Check result
        assert validation.is_valid
        assert validation.parsed_response['response'] == 2
        assert validation.parsed_response['reasoning'] == "Need more information"

    def test_validation_failure_and_retry(self):
        """Test validation failure triggers retry."""
        question = "Choose option:"
        options = ["A", "B", "C"]
        n_options = len(options)

        # First response is invalid
        invalid_response = "I think B is best"
        validation1 = ResponseValidator.validate_response(
            invalid_response, n_options, strict=True
        )
        assert not validation1.is_valid

        # Create retry prompt
        retry_prompt = ResponseValidator.create_retry_prompt(
            original_prompt=format_prompt_with_options(question, options),
            invalid_response=invalid_response,
            error_message=validation1.error_message,
            n_options=n_options
        )
        assert retry_prompt is not None

        # Second response is valid
        valid_response = '{"response": 2, "reasoning": "B is optimal"}'
        validation2 = ResponseValidator.validate_response(
            valid_response, n_options, strict=True
        )
        assert validation2.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
