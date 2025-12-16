"""Tests for structured synthesis functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agorai.synthesis.core import Agent, Council, synthesize_structured
from agorai.synthesis.validation import StructuredResponse


class TestAgentStructuredGeneration:
    """Test suite for Agent.generate_structured method."""

    @patch('agorai.synthesis.providers.get_provider')
    def test_valid_structured_response_first_try(self, mock_get_provider):
        """Test agent returns valid structured response on first attempt."""
        # Setup mock provider
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': '{"response": 2, "reasoning": "Best choice"}',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        # Create agent
        agent = Agent("test", "test-model", max_retries=2)

        # Generate structured response
        result = agent.generate_structured(
            question="Which option?",
            options=["A", "B", "C"]
        )

        # Verify result
        assert result['structured']['response'] == 2
        assert result['structured']['reasoning'] == "Best choice"
        assert result['retries'] == 0
        assert result['validation'].is_valid

        # Verify provider was called once
        assert mock_provider.generate.call_count == 1

    @patch('agorai.synthesis.providers.get_provider')
    def test_invalid_response_triggers_retry(self, mock_get_provider):
        """Test invalid response triggers retry with correction prompt."""
        # Setup mock provider with invalid then valid response
        mock_provider = Mock()
        mock_provider.generate.side_effect = [
            {'text': 'Invalid format response', 'metadata': {}},
            {'text': '{"response": 1, "reasoning": "Corrected"}', 'metadata': {}}
        ]
        mock_get_provider.return_value = mock_provider

        # Create agent
        agent = Agent("test", "test-model", max_retries=2)

        # Generate structured response
        result = agent.generate_structured(
            question="Which option?",
            options=["A", "B", "C"],
            strict=True
        )

        # Verify result
        assert result['structured']['response'] == 1
        assert result['structured']['reasoning'] == "Corrected"
        assert result['retries'] == 1

        # Verify provider was called twice
        assert mock_provider.generate.call_count == 2

        # Verify second call includes retry/correction prompt
        second_call_prompt = mock_provider.generate.call_args_list[1][0][0]
        assert "previous response" in second_call_prompt.lower()
        assert "required format" in second_call_prompt.lower()

    @patch('agorai.synthesis.providers.get_provider')
    def test_max_retries_exhausted_raises_error(self, mock_get_provider):
        """Test that exhausting max retries raises ValueError."""
        # Setup mock provider that always returns invalid format
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': 'Always invalid format',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        # Create agent with max_retries=1
        agent = Agent("test", "test-model", max_retries=1)

        # Should raise ValueError after retries exhausted
        with pytest.raises(ValueError) as exc_info:
            agent.generate_structured(
                question="Which option?",
                options=["A", "B"],
                strict=True
            )

        assert "failed to provide valid structured response" in str(exc_info.value).lower()
        assert "after 1 retries" in str(exc_info.value)

        # Verify provider was called max_retries + 1 times
        assert mock_provider.generate.call_count == 2  # initial + 1 retry

    @patch('agorai.synthesis.providers.get_provider')
    def test_lenient_mode_parsing(self, mock_get_provider):
        """Test lenient mode extracts response from informal text."""
        # Setup mock provider with informal response
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': 'I choose option 2 because it works best',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        # Create agent
        agent = Agent("test", "test-model")

        # Generate with lenient mode
        result = agent.generate_structured(
            question="Which option?",
            options=["A", "B", "C"],
            strict=False
        )

        # Verify result
        assert result['structured']['response'] == 2
        assert result['retries'] == 0

    @patch('agorai.synthesis.providers.get_provider')
    def test_context_included_in_prompt(self, mock_get_provider):
        """Test that context is included in the formatted prompt."""
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': '{"response": 1, "reasoning": "Context considered"}',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        agent = Agent("test", "test-model")

        context = "Budget: $5000, Timeline: 2 weeks"
        result = agent.generate_structured(
            question="Which option?",
            options=["A", "B"],
            context=context
        )

        # Verify context was included in prompt
        call_prompt = mock_provider.generate.call_args[0][0]
        assert context in call_prompt
        assert "CONTEXT" in call_prompt


class TestCouncilStructuredDecision:
    """Test suite for Council.decide_structured method."""

    @patch('agorai.synthesis.providers.get_provider')
    def test_basic_council_decision(self, mock_get_provider):
        """Test basic council decision with structured responses."""
        # Setup mock providers for 3 agents
        mock_provider1 = Mock()
        mock_provider1.generate.return_value = {
            'text': '{"response": 1, "reasoning": "Agent 1 choice"}',
            'metadata': {}
        }

        mock_provider2 = Mock()
        mock_provider2.generate.return_value = {
            'text': '{"response": 2, "reasoning": "Agent 2 choice"}',
            'metadata': {}
        }

        mock_provider3 = Mock()
        mock_provider3.generate.return_value = {
            'text': '{"response": 1, "reasoning": "Agent 3 choice"}',
            'metadata': {}
        }

        mock_get_provider.side_effect = [mock_provider1, mock_provider2, mock_provider3]

        # Create agents and council
        agents = [
            Agent("test", "model1"),
            Agent("test", "model2"),
            Agent("test", "model3")
        ]
        council = Council(agents, aggregation_method="majority")

        # Make decision
        result = council.decide_structured(
            question="Best option?",
            options=["Option A", "Option B", "Option C"]
        )

        # Verify result
        assert result['decision'] == "Option A"  # 2 votes for option 1
        assert result['decision_index'] == 1
        assert len(result['agent_outputs']) == 3
        assert result['total_retries'] == 0

        # Verify agent outputs
        assert result['agent_outputs'][0]['response_number'] == 1
        assert result['agent_outputs'][1]['response_number'] == 2
        assert result['agent_outputs'][2]['response_number'] == 1

    @patch('agorai.synthesis.providers.get_provider')
    def test_council_with_retries(self, mock_get_provider):
        """Test council decision where some agents need retries."""
        # Agent 1: valid first try
        mock_provider1 = Mock()
        mock_provider1.generate.return_value = {
            'text': '{"response": 1, "reasoning": "Clear choice"}',
            'metadata': {}
        }

        # Agent 2: needs retry
        mock_provider2 = Mock()
        mock_provider2.generate.side_effect = [
            {'text': 'Invalid format', 'metadata': {}},
            {'text': '{"response": 2, "reasoning": "After retry"}', 'metadata': {}}
        ]

        mock_get_provider.side_effect = [mock_provider1, mock_provider2]

        agents = [
            Agent("test", "model1", max_retries=2),
            Agent("test", "model2", max_retries=2)
        ]
        council = Council(agents, aggregation_method="majority")

        result = council.decide_structured(
            question="Choose:",
            options=["A", "B"]
        )

        # Verify retries were tracked
        assert result['total_retries'] == 1
        assert result['agent_outputs'][0]['retries'] == 0
        assert result['agent_outputs'][1]['retries'] == 1

    @patch('agorai.synthesis.providers.get_provider')
    def test_council_fails_if_agent_fails(self, mock_get_provider):
        """Test council raises error if any agent fails validation."""
        # Agent always returns invalid
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': 'Always invalid',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        agent = Agent("test", "model", max_retries=1)
        council = Council([agent], aggregation_method="majority")

        with pytest.raises(ValueError) as exc_info:
            council.decide_structured(
                question="Choose:",
                options=["A", "B"]
            )

        assert "failed to provide valid structured response" in str(exc_info.value).lower()

    @patch('agorai.synthesis.providers.get_provider')
    def test_aggregation_uses_binary_utilities(self, mock_get_provider):
        """Test that aggregation receives binary utilities based on choices."""
        # Setup agents with different choices
        responses = [
            '{"response": 1, "reasoning": "A1"}',
            '{"response": 2, "reasoning": "A2"}',
            '{"response": 1, "reasoning": "A3"}',
        ]

        mock_providers = []
        for resp in responses:
            mock_p = Mock()
            mock_p.generate.return_value = {'text': resp, 'metadata': {}}
            mock_providers.append(mock_p)

        mock_get_provider.side_effect = mock_providers

        agents = [Agent("test", f"model{i}") for i in range(3)]
        council = Council(agents, aggregation_method="majority")

        result = council.decide_structured(
            question="Choose:",
            options=["A", "B", "C"]
        )

        # Verify aggregation result
        assert result['decision'] == "A"  # Option 1 has 2 votes
        assert result['aggregation']['winner'] == 0  # Index 0

    @patch('agorai.synthesis.providers.get_provider')
    def test_empty_options_raises_error(self, mock_get_provider):
        """Test that empty options list raises ValueError."""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        agent = Agent("test", "model")
        council = Council([agent])

        with pytest.raises(ValueError) as exc_info:
            council.decide_structured(
                question="Choose:",
                options=[]
            )

        assert "must be a non-empty list" in str(exc_info.value)


class TestSynthesizeStructured:
    """Test suite for synthesize_structured function."""

    @patch('agorai.synthesis.providers.get_provider')
    def test_synthesize_structured_basic(self, mock_get_provider):
        """Test basic synthesize_structured functionality."""
        # Setup mocks
        mock_providers = []
        for i in range(2):
            mock_p = Mock()
            mock_p.generate.return_value = {
                'text': f'{{"response": 1, "reasoning": "Agent {i}"}}',
                'metadata': {}
            }
            mock_providers.append(mock_p)

        mock_get_provider.side_effect = mock_providers

        # Call synthesize_structured
        agent_configs = [
            {'provider': 'test', 'model': 'model1'},
            {'provider': 'test', 'model': 'model2'}
        ]

        result = synthesize_structured(
            question="Best option?",
            options=["A", "B", "C"],
            agents=agent_configs,
            aggregation_method="majority"
        )

        # Verify result
        assert result['decision'] == "A"
        assert len(result['agent_outputs']) == 2
        assert 'total_retries' in result

    @patch('agorai.synthesis.providers.get_provider')
    def test_synthesize_structured_with_context(self, mock_get_provider):
        """Test synthesize_structured with context."""
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': '{"response": 2, "reasoning": "With context"}',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        result = synthesize_structured(
            question="Choose:",
            options=["A", "B"],
            agents=[{'provider': 'test', 'model': 'model'}],
            context="Important context information",
            aggregation_method="majority"
        )

        # Verify context was used in prompt
        call_prompt = mock_provider.generate.call_args[0][0]
        assert "Important context information" in call_prompt

    @patch('agorai.synthesis.providers.get_provider')
    def test_synthesize_structured_with_agent_objects(self, mock_get_provider):
        """Test synthesize_structured accepts Agent objects."""
        mock_provider = Mock()
        mock_provider.generate.return_value = {
            'text': '{"response": 1, "reasoning": "Test"}',
            'metadata': {}
        }
        mock_get_provider.return_value = mock_provider

        # Pass Agent objects instead of configs
        agent = Agent("test", "model")

        result = synthesize_structured(
            question="Choose:",
            options=["A", "B"],
            agents=[agent],
            aggregation_method="majority"
        )

        assert result['decision'] in ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
