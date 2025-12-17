"""Demo: Automatic Council Creation

This example demonstrates how to use the automatic council creation feature
to generate a diverse council of agents based on a question.
"""

import os
from agorai.synthesis import create_automatic_council, create_automatic_council_simple


def demo_employment_termination():
    """Example: Employee termination decision."""
    print("=" * 80)
    print("DEMO: Automatic Council for Employment Termination")
    print("=" * 80)

    question = """
    An employee has been accused of repeatedly violating company policy by
    sharing confidential information with competitors. The evidence is strong
    but not conclusive. Should we terminate their employment?
    """

    context = """
    - Employee has been with company for 8 years
    - Previously excellent performance reviews
    - This is their first serious violation
    - The shared information could harm competitive position
    - Employee claims it was an honest mistake
    """

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        return

    # Create automatic council
    print("\n1. Creating automatic council...")
    print(f"   Question: {question.strip()[:80]}...")

    result = create_automatic_council(
        question=question,
        context=context,
        automatic_llm_api_key=api_key,
        agent_api_key=api_key,
        aggregation_method="atkinson",  # Fair aggregation
        epsilon=1.0
    )

    council = result['council']
    perspectives = result['perspectives']
    reasoning = result['reasoning']

    print(f"\n2. Council designed with {len(perspectives)} perspectives:")
    print(f"   Reasoning: {reasoning}")
    print("\n   Perspectives:")
    for i, persp in enumerate(perspectives, 1):
        print(f"   {i}. {persp['name']}")
        print(f"      Prompt: {persp['system_prompt'][:100]}...")

    # Use the council to make a decision
    print("\n3. Asking council to make decision...")
    decision = council.decide_structured(
        question="Should we terminate the employee for policy violations?",
        options=[
            "Immediate termination",
            "Final warning with probation",
            "Suspension pending investigation",
            "No action - insufficient evidence"
        ],
        context=context
    )

    print(f"\n4. Council Decision:")
    print(f"   Decision: {decision['decision']}")
    print(f"   Confidence: {decision['confidence']:.2%}")
    print(f"   Method: {decision['method']}")

    print("\n5. Individual Agent Responses:")
    for i, output in enumerate(decision['agent_outputs'], 1):
        print(f"   {i}. {output['agent']}")
        print(f"      Choice: Option {output['choice']} - {output['selected_option']}")
        print(f"      Confidence: {output['confidence']:.2%}")
        print(f"      Reasoning: {output['reasoning'][:150]}...")

    print("\n" + "=" * 80)


def demo_budget_approval_simple():
    """Example: Budget approval using simplified API."""
    print("\n" + "=" * 80)
    print("DEMO: Automatic Council for Budget Approval (Simplified API)")
    print("=" * 80)

    question = """
    We need to decide whether to approve a $2M budget for developing a new
    product line. The product has potential but also carries significant risk.
    """

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        return

    # Use simplified API
    print("\n1. Creating automatic council (simplified)...")
    council = create_automatic_council_simple(
        question=question,
        api_key=api_key,
        aggregation_method="majority"
    )

    print(f"   Council created with {len(council.agents)} agents")
    for agent in council.agents:
        print(f"   - {agent.name}")

    # Make decision
    print("\n2. Asking council to make decision...")
    decision = council.decide_structured(
        question="Should we approve the $2M budget for the new product line?",
        options=[
            "Approve full budget",
            "Approve partial budget for pilot",
            "Reject - too risky",
            "Defer decision pending market research"
        ]
    )

    print(f"\n3. Council Decision:")
    print(f"   Decision: {decision['decision']}")
    print(f"   Confidence: {decision['confidence']:.2%}")

    print("\n" + "=" * 80)


def demo_with_ollama():
    """Example: Using Ollama for local LLM deployment."""
    print("\n" + "=" * 80)
    print("DEMO: Automatic Council with Ollama (Local LLMs)")
    print("=" * 80)

    question = "Should we implement a 4-day work week in our company?"

    print("\n1. Creating automatic council with Ollama...")
    print("   (Make sure Ollama is running with llama3.2 model)")

    try:
        result = create_automatic_council(
            question=question,
            automatic_llm_provider="ollama",
            automatic_llm_model="llama3.2",
            automatic_llm_base_url="http://localhost:11434",
            agent_provider="ollama",
            agent_model="llama3.2",
            agent_base_url="http://localhost:11434",
            aggregation_method="consensus"
        )

        council = result['council']
        print(f"\n2. Council created with {len(council.agents)} agents:")
        for agent in council.agents:
            print(f"   - {agent.name}")

        # Make decision
        decision = council.decide_structured(
            question="Should we implement a 4-day work week?",
            options=[
                "Yes - implement company-wide",
                "Yes - pilot with one department",
                "No - maintain 5-day week",
                "Further research needed"
            ]
        )

        print(f"\n3. Council Decision: {decision['decision']}")
        print(f"   Confidence: {decision['confidence']:.2%}")

    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure Ollama is running: ollama serve")
        print("   And model is pulled: ollama pull llama3.2")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n🤖 Automatic Council Creation Demo\n")

    # Run demos
    demo_employment_termination()

    demo_budget_approval_simple()

    # Uncomment to test with Ollama
    # demo_with_ollama()

    print("\n✓ All demos completed!\n")
