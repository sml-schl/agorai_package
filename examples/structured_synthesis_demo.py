"""Demo: Structured Opinion Synthesis with Validation and Retry.

This example demonstrates the new structured response format with:
- Numbered options that agents must choose from
- Automatic validation of response format
- Retry mechanism for malformed responses
- Clear reasoning extraction for each agent's choice
"""

from agorai import Agent, Council, synthesize_structured


def demo_basic_structured_synthesis():
    """Basic example of structured synthesis."""
    print("=" * 70)
    print("DEMO 1: Basic Structured Synthesis")
    print("=" * 70)

    # Define question and options
    question = "Should we implement the new feature?"
    options = [
        "Approve - implement immediately",
        "Reject - not worth the effort",
        "Defer - revisit in next quarter"
    ]

    # Create agents (using Ollama for local testing)
    agents = [
        Agent("ollama", "llama3.2", name="Technical Lead", max_retries=2),
        Agent("ollama", "llama3.2", name="Product Manager", max_retries=2),
        Agent("ollama", "llama3.2", name="UX Designer", max_retries=2)
    ]

    # Create council
    council = Council(agents, aggregation_method="majority")

    # Make structured decision
    print(f"\nQuestion: {question}")
    print("\nOptions:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    print("\n" + "-" * 70)
    print("Collecting agent opinions...")
    print("-" * 70)

    result = council.decide_structured(
        question=question,
        options=options,
        strict=True
    )

    # Display results
    print("\nAgent Responses:")
    for agent_output in result['agent_outputs']:
        print(f"\n{agent_output['agent']}:")
        print(f"  Choice: Option {agent_output['response_number']} - {options[agent_output['response_number']-1]}")
        print(f"  Reasoning: {agent_output['reasoning']}")
        print(f"  Retries needed: {agent_output['retries']}")

    print("\n" + "-" * 70)
    print("COLLECTIVE DECISION")
    print("-" * 70)
    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Aggregation Method: {result['method']}")
    print(f"Total Retries: {result['total_retries']}")


def demo_with_context():
    """Example with additional context."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Structured Synthesis with Context")
    print("=" * 70)

    question = "Which marketing channel should we prioritize?"
    options = [
        "Social Media",
        "Email Marketing",
        "Content Marketing",
        "Paid Advertising"
    ]

    context = """
    Current Situation:
    - Budget: $50,000
    - Timeline: Q1 2024
    - Target: Young professionals (25-35)
    - Current reach: 10,000 followers
    - Goal: 50% increase in conversions
    """

    # Using config dicts for agents
    agent_configs = [
        {'provider': 'ollama', 'model': 'llama3.2', 'name': 'Marketing Director'},
        {'provider': 'ollama', 'model': 'llama3.2', 'name': 'Data Analyst'},
        {'provider': 'ollama', 'model': 'llama3.2', 'name': 'Brand Manager'}
    ]

    print(f"\nQuestion: {question}")
    print(f"\nContext:{context}")
    print("\nOptions:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    result = synthesize_structured(
        question=question,
        options=options,
        agents=agent_configs,
        context=context,
        aggregation_method="atkinson",
        epsilon=0.5  # Moderate inequality aversion
    )

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Winning Strategy: {result['decision']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nDetailed Agent Reasoning:")
    for agent_output in result['agent_outputs']:
        print(f"\n{agent_output['agent']} → Option {agent_output['response_number']}")
        print(f"  {agent_output['reasoning'][:150]}...")


def demo_different_aggregation_methods():
    """Compare different aggregation methods."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Comparing Aggregation Methods")
    print("=" * 70)

    question = "Which technology stack should we use?"
    options = [
        "Python/Django",
        "Node.js/Express",
        "Java/Spring",
        "Go/Gin"
    ]

    agents = [
        Agent("ollama", "llama3.2", name="Backend Engineer"),
        Agent("ollama", "llama3.2", name="DevOps Engineer"),
        Agent("ollama", "llama3.2", name="Tech Lead")
    ]

    # Test different aggregation methods
    methods = ["majority", "borda", "maximin"]

    for method in methods:
        print(f"\n{'-' * 70}")
        print(f"Aggregation Method: {method.upper()}")
        print('-' * 70)

        council = Council(agents, aggregation_method=method)

        result = council.decide_structured(
            question=question,
            options=options
        )

        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.2%}")

        # Show vote distribution
        votes = {}
        for agent_output in result['agent_outputs']:
            choice = agent_output['response_number']
            votes[choice] = votes.get(choice, 0) + 1

        print("Vote Distribution:")
        for opt_num in sorted(votes.keys()):
            print(f"  Option {opt_num}: {votes[opt_num]} vote(s)")


def demo_error_handling():
    """Demonstrate validation and retry mechanism."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Validation and Retry Mechanism")
    print("=" * 70)

    # This demo shows how the system handles potential formatting issues
    # In practice, LLMs usually follow instructions well, but the retry
    # mechanism ensures robustness

    question = "Is this feature ready for production?"
    options = ["Yes", "No", "Needs more testing"]

    agent = Agent("ollama", "llama3.2", max_retries=3)

    print(f"\nQuestion: {question}")
    print("\nOptions:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    print("\n" + "-" * 70)
    print("Agent Configuration:")
    print(f"  Max Retries: {agent.max_retries}")
    print(f"  Validation: Strict regex-based format checking")
    print("-" * 70)

    try:
        result = agent.generate_structured(
            question=question,
            options=options,
            strict=True
        )

        print("\nResponse received successfully!")
        print(f"  Choice: Option {result['structured']['response']}")
        print(f"  Reasoning: {result['structured']['reasoning']}")
        print(f"  Retries used: {result['retries']}")

        if result['retries'] > 0:
            print(f"\n  ⚠ Agent needed {result['retries']} retry(ies) to format response correctly")
        else:
            print("\n  ✓ Agent provided correctly formatted response on first try")

    except ValueError as e:
        print(f"\n❌ Validation failed after all retries: {e}")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("AGORAI STRUCTURED SYNTHESIS DEMONSTRATION")
    print("*" * 70)
    print("\nThis demo showcases the enhanced agorai package with:")
    print("  • Structured response format with numbered options")
    print("  • Automatic validation using regex patterns")
    print("  • Retry mechanism for malformed responses")
    print("  • Clear reasoning extraction")
    print("  • Robust mathematical aggregation")

    try:
        # Run demos
        demo_basic_structured_synthesis()
        demo_with_context()
        demo_different_aggregation_methods()
        demo_error_handling()

        print("\n\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print("\nNote: Make sure Ollama is running locally with llama3.2 model")
        print("Install with: ollama pull llama3.2")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
