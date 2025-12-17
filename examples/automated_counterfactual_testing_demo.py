"""Demo: Automated Counterfactual Testing for Causal Robustness Evaluation.

This script demonstrates how to use the automated counterfactual testing
pipeline to evaluate causal robustness of bias detection systems.

The pipeline:
1. Detects protected attributes in images (ethnicity, gender, age, etc.)
2. Generates counterfactual images by systematically modifying these attributes
3. Evaluates model predictions on original vs counterfactual images
4. Identifies spurious correlations (when changing irrelevant attributes affects predictions)

Requirements:
- OpenAI API key (for GPT-4V detection and DALL-E 3 generation)
- Or: Replicate API key (for Stable Diffusion generation)
"""

import os
import yaml
from pathlib import Path

from agorai import AgentCouncil, OpenAIAgent, AgentConfig
from agorai.testing import (
    create_tester_from_config,
    AutomatedTestingConfig,
)


def load_config(config_path: str = "../config/counterfactual_testing_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_example_council():
    """Create an example AgentCouncil for bias detection.

    In production, this would be your actual bias detection system
    with multiple diverse agents/perspectives.
    """
    # Create agents representing different cultural perspectives
    western_agent = OpenAIAgent(AgentConfig(
        name="Western",
        model="gpt-4-vision-preview",
        system_prompt="""You are analyzing images for hateful content from a Western cultural perspective.
        Consider Western norms around hate speech, discrimination, and offensive content.""",
        temperature=0.0
    ))

    eastern_agent = OpenAIAgent(AgentConfig(
        name="Eastern",
        model="gpt-4-vision-preview",
        system_prompt="""You are analyzing images for hateful content from an Eastern cultural perspective.
        Consider Eastern norms around respect, harmony, and offensive content.""",
        temperature=0.0
    ))

    global_south_agent = OpenAIAgent(AgentConfig(
        name="GlobalSouth",
        model="gpt-4-vision-preview",
        system_prompt="""You are analyzing images for hateful content from a Global South perspective.
        Consider postcolonial contexts, representation, and discrimination.""",
        temperature=0.0
    ))

    # Create council
    council = AgentCouncil(
        agents=[western_agent, eastern_agent, global_south_agent],
        cfg=None  # Will use default configuration
    )

    return council


def run_single_test_example(config_path: str):
    """Example: Running a single counterfactual test."""
    print("="*70)
    print("EXAMPLE 1: Single Image Counterfactual Test")
    print("="*70)

    # Load configuration
    config = load_config(config_path)

    # Create council
    council = create_example_council()

    # Create automated tester
    tester = create_tester_from_config(council, config)

    # Run test on a single image
    # NOTE: Replace with actual image URL
    result = tester.run_test_case(
        test_case_id="example_001",
        image_url="https://example.com/test_image.jpg",
        text_content="Example meme caption",
        ground_truth_label="not_hateful"  # Optional
    )

    # Print results
    print(f"\nTest Results:")
    print(f"  Test ID: {result.test_id}")
    print(f"  Protected Attributes Detected: {len(result.protected_attributes)}")
    for attr in result.protected_attributes:
        print(f"    - {attr.attribute_type.value}: {attr.value} (confidence: {attr.confidence:.2f})")

    print(f"\n  Original Prediction: {result.original_prediction.get('winner')}")
    print(f"  Counterfactuals Generated: {len(result.counterfactuals)}")
    print(f"  Consistency Score: {result.consistency_score:.2%}")
    print(f"  Spurious Correlation Detected: {result.spurious_correlation_detected}")

    if result.spurious_correlation_detected:
        print("\n  ⚠️  Warning: Spurious correlations found!")
        print("  The model's predictions change when irrelevant protected attributes are modified.")
        print("  This suggests the model may be relying on biased shortcuts rather than causal reasoning.")

    return result


def run_batch_test_example(config_path: str):
    """Example: Running batch counterfactual tests."""
    print("\n\n")
    print("="*70)
    print("EXAMPLE 2: Batch Counterfactual Testing")
    print("="*70)

    # Load configuration
    config = load_config(config_path)

    # Create council
    council = create_example_council()

    # Create automated tester
    tester = create_tester_from_config(council, config)

    # Define batch of test cases
    # NOTE: Replace with actual image URLs from your dataset
    test_cases = [
        {
            "id": "hateful_memes_001",
            "image_url": "https://example.com/meme_001.jpg",
            "text_content": "Sample text 1",
            "ground_truth": "hateful"
        },
        {
            "id": "hateful_memes_002",
            "image_url": "https://example.com/meme_002.jpg",
            "text_content": "Sample text 2",
            "ground_truth": "not_hateful"
        },
        # Add more test cases...
    ]

    # Run batch tests
    results = tester.run_batch_tests(test_cases)

    # Compute aggregate statistics
    total_tests = len(results)
    avg_consistency = sum(r.consistency_score for r in results) / total_tests
    spurious_count = sum(1 for r in results if r.spurious_correlation_detected)

    print(f"\nBatch Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Average Consistency Score: {avg_consistency:.2%}")
    print(f"  Tests with Spurious Correlations: {spurious_count} ({spurious_count/total_tests:.1%})")

    # Export results
    output_path = "./counterfactual_test_results.json"
    tester.export_results(results, output_path)

    return results


def analyze_results_example():
    """Example: Analyzing counterfactual test results."""
    print("\n\n")
    print("="*70)
    print("EXAMPLE 3: Analyzing Counterfactual Test Results")
    print("="*70)

    # Load results from previous run
    import json
    with open("./counterfactual_test_results.json", 'r') as f:
        results = json.load(f)

    # Analyze which attributes most frequently cause prediction changes
    attribute_impact = {}

    for test_result in results:
        for cf in test_result.get("counterfactuals", []):
            attr_modified = cf.get("attribute_modified")
            prediction = cf.get("prediction", {})
            original_pred = test_result.get("original_prediction", {})

            # Check if prediction changed
            if prediction.get("winner") != original_pred.get("winner"):
                attribute_impact[attr_modified] = attribute_impact.get(attr_modified, 0) + 1

    print("\nAttribute Impact Analysis:")
    print("(How often does changing each attribute type cause prediction changes?)\n")

    sorted_attrs = sorted(attribute_impact.items(), key=lambda x: x[1], reverse=True)
    for attr, count in sorted_attrs:
        print(f"  {attr}: {count} prediction changes")

    print("\nInterpretation:")
    print("  - Attributes with high counts may indicate spurious correlations")
    print("  - The model may be using these attributes as shortcuts")
    print("  - Consider investigating why these attributes affect predictions")


def custom_variation_example():
    """Example: Creating custom counterfactual variations."""
    print("\n\n")
    print("="*70)
    print("EXAMPLE 4: Custom Counterfactual Variations")
    print("="*70)

    from agorai.testing import (
        ProtectedAttribute,
        ProtectedAttributeType,
        create_counterfactual_variation,
        ModificationStrategy,
    )

    # Create a detected attribute
    detected_attr = ProtectedAttribute(
        attribute_type=ProtectedAttributeType.ETHNICITY,
        value="african",
        confidence=0.9,
        description="Person with dark skin tone"
    )

    # Create custom variations with different strategies
    variations = [
        # Swap to specific alternative
        create_counterfactual_variation(
            attribute=detected_attr,
            target_value="european",
            strategy=ModificationStrategy.SWAP,
            preserve_elements=["text", "background", "pose"]
        ),

        # Remove attribute entirely
        create_counterfactual_variation(
            attribute=detected_attr,
            target_value="",  # Empty for removal
            strategy=ModificationStrategy.REMOVE,
            preserve_elements=["text", "background", "composition"]
        ),

        # Neutralize (make ambiguous)
        create_counterfactual_variation(
            attribute=detected_attr,
            target_value="neutral",
            strategy=ModificationStrategy.NEUTRALIZE,
            preserve_elements=["text", "background"]
        ),
    ]

    print("Created custom variations:")
    for i, var in enumerate(variations, 1):
        print(f"\n  Variation {i}:")
        print(f"    Strategy: {var.modification_strategy.value}")
        print(f"    Original: {var.original_attribute.value}")
        print(f"    Target: {var.target_value}")
        print(f"    Preserve: {', '.join(var.preserve_elements)}")


def main():
    """Run all examples."""
    config_path = "../config/counterfactual_testing_config.yaml"

    # Check if config exists
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        print("Please create the configuration file first.")
        return

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run these examples.")
        return

    print("\n" + "="*70)
    print("AUTOMATED COUNTERFACTUAL TESTING - DEMO")
    print("="*70)
    print("\nThis demo shows how to use automated counterfactual testing")
    print("to evaluate causal robustness of bias detection systems.\n")

    # Run examples
    try:
        # Example 1: Single test
        single_result = run_single_test_example(config_path)

        # Example 2: Batch test (commented out to avoid API costs)
        # batch_results = run_batch_test_example(config_path)

        # Example 3: Analysis (requires previous results)
        # analyze_results_example()

        # Example 4: Custom variations
        custom_variation_example()

        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("  1. Replace example URLs with your actual test images")
        print("  2. Run batch tests on your full dataset")
        print("  3. Analyze results to identify spurious correlations")
        print("  4. Use insights to improve your bias detection system")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("\nCommon issues:")
        print("  - Check that your API key is valid")
        print("  - Verify image URLs are accessible")
        print("  - Ensure sufficient API credits/quota")


if __name__ == "__main__":
    main()
