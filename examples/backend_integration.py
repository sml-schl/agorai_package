"""
Example: Backend Integration with AgorAI Package

This example demonstrates how to use the agorai PyPI package in your backend.
It shows the three main use cases:
1. Pure aggregation (no LLM)
2. LLM-based synthesis
3. Full bias mitigation
"""

# ============================================================================
# Example 1: Pure Mathematical Aggregation
# ============================================================================

def example_pure_aggregation():
    """Example of using aggregation without any LLM dependencies."""
    from agorai.aggregate import aggregate, list_methods

    print("="*60)
    print("EXAMPLE 1: Pure Mathematical Aggregation")
    print("="*60)

    # Scenario: 3 agents rating 2 policy options
    # utilities[i][j] = agent i's utility for option j
    utilities = [
        [0.9, 0.3],  # Agent 1 strongly prefers option 0
        [0.4, 0.8],  # Agent 2 prefers option 1
        [0.7, 0.5],  # Agent 3 slightly prefers option 0
    ]

    # Try different aggregation methods
    methods = ["majority", "atkinson", "maximin", "borda"]

    for method in methods:
        result = aggregate(utilities, method=method, epsilon=1.0 if method == "atkinson" else None)
        winner = result["winner"]
        scores = result["scores"]

        print(f"\nMethod: {method}")
        print(f"  Winner: Option {winner}")
        print(f"  Scores: {scores}")

    # List all available methods
    print("\n\nAll available aggregation methods:")
    all_methods = list_methods()
    for m in all_methods[:5]:  # Show first 5
        print(f"  - {m['name']} ({m['category']}): {m['description']}")


# ============================================================================
# Example 2: LLM-Based Opinion Synthesis
# ============================================================================

def example_llm_synthesis():
    """Example of using LLM agents for opinion synthesis."""
    from agorai.synthesis import Agent, synthesize

    print("\n\n" + "="*60)
    print("EXAMPLE 2: LLM-Based Opinion Synthesis")
    print("="*60)

    # Create diverse agents
    # NOTE: This requires Ollama running locally or API keys for cloud providers
    agents = [
        Agent(
            provider="ollama",
            model="llama3.2",
            system_prompt="You are a security expert. Prioritize safety.",
            name="Security-Expert"
        ),
        Agent(
            provider="ollama",
            model="llama3.2",
            system_prompt="You are a user experience expert. Prioritize usability.",
            name="UX-Expert"
        ),
        Agent(
            provider="ollama",
            model="llama3.2",
            system_prompt="You are a business analyst. Prioritize ROI.",
            name="Business-Analyst"
        ),
    ]

    # Define decision options
    candidates = ["approve", "reject", "revise"]

    # Make a collective decision
    prompt = """
    Should we ship the new feature that allows users to upload custom profile pictures?

    Consider:
    - Security implications
    - User experience
    - Business value

    Respond with your recommendation and brief reasoning.
    """

    try:
        result = synthesize(
            prompt=prompt,
            agents=agents,
            candidates=candidates,
            aggregation_method="majority"
        )

        print(f"\nCollective Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nIndividual Agent Perspectives:")
        for output in result['agent_outputs']:
            print(f"\n  {output['agent']}:")
            print(f"    {output['text'][:100]}...")

    except Exception as e:
        print(f"\nNote: LLM synthesis requires Ollama or API keys")
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull model: ollama pull llama3.2")
        print("  3. Run Ollama server: ollama serve")


# ============================================================================
# Example 3: Bias Mitigation Pipeline
# ============================================================================

def example_bias_mitigation():
    """Example of using the full bias mitigation pipeline."""
    from agorai.bias import mitigate_bias, BiasConfig

    print("\n\n" + "="*60)
    print("EXAMPLE 3: Bias Mitigation Pipeline")
    print("="*60)

    # Configure bias mitigation
    config = BiasConfig(
        context="hate_speech_detection",
        providers=["ollama"],  # Use local Ollama
        aggregation_method="atkinson",  # Fairness-aware aggregation
        cultural_perspectives=5,  # 5 diverse cultural viewpoints
        rounds=2,  # 2 rounds of deliberation
        temperature=0.7
    )

    # Test content
    test_cases = [
        "This is a normal friendly message",
        "I strongly disagree with that policy decision",
        # Add more test cases as needed
    ]

    for i, content in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Content: {content}")

        try:
            result = mitigate_bias(
                input_text=content,
                candidates=["hateful", "borderline", "not_hateful"],
                config=config
            )

            print(f"\nBias-Mitigated Decision: {result['decision']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nFairness Metrics:")
            for metric, value in result['fairness_metrics'].items():
                print(f"  {metric}: {value}")

        except Exception as e:
            print(f"\nNote: Bias mitigation requires LLM access")
            print(f"Error: {e}")
            break  # Don't repeat error for each test case


# ============================================================================
# Example 4: Backend Service Integration
# ============================================================================

class ContentModerationService:
    """Example backend service using agorai for content moderation."""

    def __init__(self, aggregation_method="schulze_condorcet"):
        from agorai.aggregate import aggregate, list_methods
        self.aggregate = aggregate
        self.aggregation_method = aggregation_method
        self.available_methods = [m['name'] for m in list_methods()]

    def moderate_content_simple(self, content: str, moderator_scores: list) -> dict:
        """
        Simple content moderation using pure aggregation.

        Args:
            content: Content to moderate
            moderator_scores: List of [safe_score, unsafe_score] from each moderator

        Returns:
            Moderation decision
        """
        result = self.aggregate(
            utilities=moderator_scores,
            method=self.aggregation_method
        )

        decision = "safe" if result["winner"] == 0 else "unsafe"

        return {
            "content": content,
            "decision": decision,
            "scores": result["scores"],
            "method": self.aggregation_method
        }

    def moderate_content_with_llm(self, content: str, use_bias_mitigation=False):
        """
        Content moderation using LLM agents.

        Args:
            content: Content to moderate
            use_bias_mitigation: Whether to use full bias mitigation pipeline

        Returns:
            Moderation decision with bias analysis
        """
        if use_bias_mitigation:
            from agorai.bias import mitigate_bias, BiasConfig

            config = BiasConfig(
                context="content_moderation",
                providers=["ollama"],
                aggregation_method=self.aggregation_method,
                cultural_perspectives=5
            )

            result = mitigate_bias(
                input_text=content,
                config=config
            )

            return {
                "content": content,
                "decision": result["decision"],
                "confidence": result["confidence"],
                "fairness_metrics": result["fairness_metrics"],
                "bias_mitigated": True
            }
        else:
            from agorai.synthesis import Agent, synthesize

            # Create simple moderation agents
            agents = [
                Agent("ollama", "llama3.2", name="Moderator-1"),
                Agent("ollama", "llama3.2", name="Moderator-2"),
            ]

            result = synthesize(
                prompt=f"Is this content safe for our platform? Content: {content}",
                agents=agents,
                candidates=["safe", "unsafe", "review_needed"],
                aggregation_method=self.aggregation_method
            )

            return {
                "content": content,
                "decision": result["decision"],
                "confidence": result["confidence"],
                "bias_mitigated": False
            }

def example_backend_service():
    """Example of integrating agorai into a backend service."""
    print("\n\n" + "="*60)
    print("EXAMPLE 4: Backend Service Integration")
    print("="*60)

    # Initialize service
    service = ContentModerationService(aggregation_method="atkinson")

    print("\nSimple moderation (no LLM, just aggregation):")
    # Simulated moderator scores: [safe_score, unsafe_score]
    moderator_scores = [
        [0.9, 0.1],  # Moderator 1: likely safe
        [0.8, 0.2],  # Moderator 2: likely safe
        [0.3, 0.7],  # Moderator 3: likely unsafe
    ]

    result = service.moderate_content_simple(
        content="Example content",
        moderator_scores=moderator_scores
    )

    print(f"  Decision: {result['decision']}")
    print(f"  Scores: {result['scores']}")
    print(f"  Method: {result['method']}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" AgorAI Package - Backend Integration Examples")
    print("="*60)

    # Example 1: Always works (no dependencies)
    example_pure_aggregation()

    # Example 2: Requires Ollama or API keys
    print("\n\nNOTE: Examples 2-4 require LLM access (Ollama or cloud APIs)")
    user_input = input("\nRun LLM examples? (requires Ollama running) [y/N]: ")

    if user_input.lower() == 'y':
        example_llm_synthesis()
        example_bias_mitigation()
        example_backend_service()
    else:
        print("\nSkipping LLM examples. Run with Ollama to see full functionality.")

    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Install the package: pip install agorai[all]")
    print("  2. Review documentation: docs/")
    print("  3. Check migration guide: BACKEND_MIGRATION_GUIDE.md")
    print("  4. Integrate into your backend!")
