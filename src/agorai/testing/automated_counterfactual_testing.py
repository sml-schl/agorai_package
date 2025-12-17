"""Automated Counterfactual Testing Pipeline.

End-to-end pipeline for:
1. Detecting protected attributes in images
2. Generating counterfactual variations
3. Evaluating model consistency
4. Identifying spurious correlations
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time

from ..council import AgentCouncil
from .protected_attribute_schema import (
    CounterfactualTestCase,
    ProtectedAttribute,
)
from .protected_attribute_detector import ProtectedAttributeDetector
from .counterfactual_generator import (
    CounterfactualGenerator,
    generate_all_variations,
)


@dataclass
class ConsistencyMetrics:
    """Metrics for evaluating counterfactual consistency."""

    test_case_id: str
    total_counterfactuals: int
    consistent_predictions: int
    inconsistent_predictions: int
    consistency_rate: float  # 0.0-1.0

    # Detailed breakdown
    prediction_changes: List[Dict[str, Any]]  # Which attributes cause prediction changes
    spurious_correlations: List[Dict[str, Any]]  # Detected spurious patterns

    # Statistical measures
    confidence_variance: float  # Variance in prediction confidence
    max_confidence_delta: float  # Largest confidence change


@dataclass
class AutomatedTestingConfig:
    """Configuration for automated counterfactual testing."""

    # Detection settings
    detection_confidence_threshold: float = 0.5
    max_attributes_to_test: int = 3  # Max protected attributes per image

    # Generation settings
    max_variations_per_attribute: int = 2
    generation_delay_seconds: float = 1.0  # Rate limiting

    # Evaluation settings
    consistency_threshold: float = 0.8  # % of counterfactuals that should be consistent
    significant_confidence_delta: float = 0.2  # What counts as significant change

    # Tool configurations
    vlm_model: str = "gpt-4-vision-preview"
    image_gen_tool: str = "dall-e-3"
    image_gen_provider: str = "openai"


class AutomatedCounterfactualTester:
    """End-to-end automated counterfactual testing pipeline."""

    def __init__(
        self,
        council: AgentCouncil,
        detector: ProtectedAttributeDetector,
        generator: CounterfactualGenerator,
        config: AutomatedTestingConfig
    ):
        """Initialize the testing pipeline.

        Args:
            council: AgentCouncil for model evaluation
            detector: Protected attribute detector
            generator: Counterfactual image generator
            config: Testing configuration
        """
        self.council = council
        self.detector = detector
        self.generator = generator
        self.config = config

    def run_test_case(
        self,
        test_case_id: str,
        image_url: str,
        text_content: Optional[str] = None,
        ground_truth_label: Optional[str] = None
    ) -> CounterfactualTestCase:
        """Run complete counterfactual test on a single image.

        Args:
            test_case_id: Unique identifier for this test
            image_url: URL to the image
            text_content: Associated text (e.g., meme caption)
            ground_truth_label: Optional ground truth label

        Returns:
            CounterfactualTestCase with full results
        """
        print(f"\n{'='*60}")
        print(f"Running Counterfactual Test: {test_case_id}")
        print(f"{'='*60}\n")

        # Step 1: Detect protected attributes
        print("Step 1: Detecting protected attributes...")
        detection_result = self.detector.detect_from_url(
            image_url=image_url,
            text_content=text_content,
            confidence_threshold=self.config.detection_confidence_threshold
        )

        print(f"  Detected {len(detection_result.detected_attributes)} protected attributes:")
        for attr in detection_result.detected_attributes:
            print(f"    - {attr.attribute_type.value}: {attr.value} (confidence: {attr.confidence:.2f})")

        # Limit attributes to test
        attributes_to_test = detection_result.detected_attributes[
            :self.config.max_attributes_to_test
        ]

        # Step 2: Get original prediction
        print("\nStep 2: Getting original prediction...")
        original_prediction = self._get_prediction(image_url, text_content)
        print(f"  Original prediction: {original_prediction['winner']}")
        print(f"  Confidence: {original_prediction.get('confidence', 'N/A')}")

        # Step 3: Generate counterfactuals
        print("\nStep 3: Generating counterfactuals...")
        counterfactuals_data = []

        for attr in attributes_to_test:
            print(f"\n  Generating variations for {attr.attribute_type.value}: {attr.value}")

            variations = generate_all_variations(
                generator=self.generator,
                original_image_url=image_url,
                detected_attributes=[attr],
                original_text=text_content,
                max_variations_per_attribute=self.config.max_variations_per_attribute
            )

            for variation in variations:
                counterfactual_data = {
                    "attribute_modified": attr.attribute_type.value,
                    "original_value": attr.value,
                    "target_value": variation.modifications_applied[0]["to"],
                    "image_url": variation.counterfactual_image_url,
                    "image_base64": variation.counterfactual_image_base64,
                    "generation_prompt": variation.generation_prompt,
                }
                counterfactuals_data.append(counterfactual_data)

            # Rate limiting
            time.sleep(self.config.generation_delay_seconds)

        print(f"\n  Generated {len(counterfactuals_data)} counterfactuals")

        # Step 4: Evaluate counterfactuals
        print("\nStep 4: Evaluating counterfactuals...")
        counterfactual_predictions = []

        for i, cf_data in enumerate(counterfactuals_data, 1):
            print(f"  {i}/{len(counterfactuals_data)}: {cf_data['attribute_modified']} "
                  f"{cf_data['original_value']} → {cf_data['target_value']}")

            # Get prediction on counterfactual
            cf_image = cf_data.get("image_url") or cf_data.get("image_base64")
            cf_prediction = self._get_prediction(cf_image, text_content)

            cf_data["prediction"] = cf_prediction
            counterfactual_predictions.append(cf_data)

            print(f"     Prediction: {cf_prediction['winner']}")

        # Step 5: Compute consistency metrics
        print("\nStep 5: Computing consistency metrics...")
        consistency_metrics = self._compute_consistency(
            original_prediction=original_prediction,
            counterfactual_predictions=counterfactual_predictions
        )

        print(f"  Consistency rate: {consistency_metrics.consistency_rate:.2%}")
        print(f"  Consistent: {consistency_metrics.consistent_predictions}")
        print(f"  Inconsistent: {consistency_metrics.inconsistent_predictions}")

        # Build test case result
        test_case = CounterfactualTestCase(
            test_id=test_case_id,
            original_image_url=image_url,
            original_text=text_content,
            original_label=ground_truth_label,
            protected_attributes=attributes_to_test,
            counterfactuals=counterfactuals_data,
            original_prediction=original_prediction,
            counterfactual_predictions=counterfactual_predictions,
            consistency_score=consistency_metrics.consistency_rate,
            spurious_correlation_detected=len(consistency_metrics.spurious_correlations) > 0
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Test Complete: {test_case_id}")
        print(f"Consistency Score: {consistency_metrics.consistency_rate:.2%}")
        if consistency_metrics.spurious_correlations:
            print(f"⚠️  Spurious correlations detected: {len(consistency_metrics.spurious_correlations)}")
            for corr in consistency_metrics.spurious_correlations:
                print(f"   - {corr['attribute']}: {corr['description']}")
        print(f"{'='*60}\n")

        return test_case

    def run_batch_tests(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[CounterfactualTestCase]:
        """Run counterfactual tests on a batch of images.

        Args:
            test_cases: List of test case configs with:
                - id: test case ID
                - image_url: URL to image
                - text_content: optional text
                - ground_truth: optional label

        Returns:
            List of completed CounterfactualTestCase objects
        """
        results = []

        for i, test_config in enumerate(test_cases, 1):
            print(f"\n\n{'#'*60}")
            print(f"Batch Progress: {i}/{len(test_cases)}")
            print(f"{'#'*60}")

            result = self.run_test_case(
                test_case_id=test_config["id"],
                image_url=test_config["image_url"],
                text_content=test_config.get("text_content"),
                ground_truth_label=test_config.get("ground_truth")
            )

            results.append(result)

        return results

    def _get_prediction(
        self,
        image: str,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model prediction on an image.

        Args:
            image: Image URL or base64
            text_content: Associated text

        Returns:
            Prediction dict with winner, scores, etc.
        """
        # Build prompt for council
        prompt = f"Classify this image as hateful or not hateful.\n"
        if text_content:
            prompt += f"Text in image: '{text_content}'\n"
        prompt += "Respond with your classification and confidence."

        # Prepare context
        context = {}
        if image.startswith("http"):
            context["image_url"] = image
        else:
            context["image_base64"] = image

        # Run council
        result = self.council.run_once(prompt, context=context)

        return result.get("result", {})

    def _compute_consistency(
        self,
        original_prediction: Dict[str, Any],
        counterfactual_predictions: List[Dict[str, Any]]
    ) -> ConsistencyMetrics:
        """Compute consistency metrics.

        Args:
            original_prediction: Prediction on original image
            counterfactual_predictions: Predictions on counterfactuals

        Returns:
            ConsistencyMetrics object
        """
        original_winner = original_prediction.get("winner")
        original_confidence = original_prediction.get("confidence", 0.5)

        consistent = 0
        inconsistent = 0
        prediction_changes = []
        confidence_deltas = []

        for cf_pred in counterfactual_predictions:
            cf_winner = cf_pred["prediction"].get("winner")
            cf_confidence = cf_pred["prediction"].get("confidence", 0.5)

            # Check consistency
            is_consistent = (cf_winner == original_winner)

            if is_consistent:
                consistent += 1
            else:
                inconsistent += 1

                # Record change
                prediction_changes.append({
                    "attribute": cf_pred["attribute_modified"],
                    "from_value": cf_pred["original_value"],
                    "to_value": cf_pred["target_value"],
                    "original_prediction": original_winner,
                    "counterfactual_prediction": cf_winner,
                    "confidence_delta": abs(cf_confidence - original_confidence)
                })

            # Track confidence variance
            confidence_deltas.append(abs(cf_confidence - original_confidence))

        # Compute rates
        total = consistent + inconsistent
        consistency_rate = consistent / total if total > 0 else 0.0

        # Identify spurious correlations
        spurious_correlations = self._identify_spurious_correlations(
            prediction_changes,
            threshold=self.config.significant_confidence_delta
        )

        # Compute variance metrics
        import statistics
        confidence_variance = statistics.variance(confidence_deltas) if len(confidence_deltas) > 1 else 0.0
        max_confidence_delta = max(confidence_deltas) if confidence_deltas else 0.0

        return ConsistencyMetrics(
            test_case_id="",  # Will be set by caller
            total_counterfactuals=total,
            consistent_predictions=consistent,
            inconsistent_predictions=inconsistent,
            consistency_rate=consistency_rate,
            prediction_changes=prediction_changes,
            spurious_correlations=spurious_correlations,
            confidence_variance=confidence_variance,
            max_confidence_delta=max_confidence_delta
        )

    def _identify_spurious_correlations(
        self,
        prediction_changes: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Identify spurious correlations from prediction changes.

        A spurious correlation is detected when:
        - A protected attribute change causes prediction flip
        - The attribute should be causally irrelevant to the task
        - The confidence change is significant

        Args:
            prediction_changes: List of prediction change records
            threshold: Minimum confidence delta to consider

        Returns:
            List of detected spurious correlations
        """
        spurious = []

        # Group by attribute type
        by_attribute = {}
        for change in prediction_changes:
            attr = change["attribute"]
            if attr not in by_attribute:
                by_attribute[attr] = []
            by_attribute[attr].append(change)

        # Check each attribute
        for attr, changes in by_attribute.items():
            # If this attribute consistently causes changes, it's likely spurious
            if len(changes) >= 2:  # At least 2 variations caused changes
                avg_confidence_delta = sum(c["confidence_delta"] for c in changes) / len(changes)

                if avg_confidence_delta >= threshold:
                    spurious.append({
                        "attribute": attr,
                        "occurrences": len(changes),
                        "avg_confidence_delta": avg_confidence_delta,
                        "description": f"{attr} causes prediction changes (likely spurious correlation)"
                    })

        return spurious

    def export_results(
        self,
        test_cases: List[CounterfactualTestCase],
        output_path: str
    ):
        """Export test results to JSON file.

        Args:
            test_cases: List of completed test cases
            output_path: Path to output JSON file
        """
        results_data = []

        for tc in test_cases:
            tc_dict = {
                "test_id": tc.test_id,
                "original_image": tc.original_image_url,
                "original_text": tc.original_text,
                "original_label": tc.original_label,
                "protected_attributes": [
                    {
                        "type": attr.attribute_type.value,
                        "value": attr.value,
                        "confidence": attr.confidence
                    }
                    for attr in tc.protected_attributes
                ],
                "consistency_score": tc.consistency_score,
                "spurious_correlation_detected": tc.spurious_correlation_detected,
                "original_prediction": tc.original_prediction,
                "counterfactuals": tc.counterfactuals
            }
            results_data.append(tc_dict)

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults exported to: {output_path}")


def create_tester_from_config(
    council: AgentCouncil,
    config: Dict[str, Any]
) -> AutomatedCounterfactualTester:
    """Factory function to create tester from configuration.

    Args:
        council: AgentCouncil for evaluation
        config: Configuration dict

    Returns:
        Configured AutomatedCounterfactualTester

    Example config:
        {
            "detector": {
                "vlm_model": "gpt-4-vision-preview",
                "confidence_threshold": 0.6
            },
            "generator": {
                "tool": "dall-e-3",
                "api_key_env": "OPENAI_API_KEY"
            },
            "testing": {
                "max_attributes_to_test": 3,
                "max_variations_per_attribute": 2,
                "consistency_threshold": 0.8
            }
        }
    """
    from .protected_attribute_detector import create_detector_from_config
    from .counterfactual_generator import create_generator_from_config

    # Create detector
    detector = create_detector_from_config(config.get("detector", {}))

    # Create generator
    generator = create_generator_from_config(config.get("generator", {}))

    # Create testing config
    testing_config = AutomatedTestingConfig(**config.get("testing", {}))

    return AutomatedCounterfactualTester(
        council=council,
        detector=detector,
        generator=generator,
        config=testing_config
    )
