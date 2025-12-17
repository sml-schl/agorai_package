"""Automated Counterfactual Testing for Causal Robustness Evaluation.

This module provides tools for:
1. Detecting protected attributes in images (ethnicity, gender, age, etc.)
2. Generating counterfactual images by modifying protected attributes
3. Evaluating model consistency across counterfactuals
4. Identifying spurious correlations

Example usage:
    >>> from agorai.testing import (
    ...     create_tester_from_config,
    ...     AutomatedTestingConfig
    ... )
    >>> from agorai import AgentCouncil
    >>>
    >>> # Create council for evaluation
    >>> council = AgentCouncil(agents=[...])
    >>>
    >>> # Create automated tester
    >>> config = {
    ...     "detector": {"vlm_model": "gpt-4-vision-preview"},
    ...     "generator": {"tool": "dall-e-3"},
    ...     "testing": {"max_attributes_to_test": 3}
    ... }
    >>> tester = create_tester_from_config(council, config)
    >>>
    >>> # Run test
    >>> result = tester.run_test_case(
    ...     test_case_id="test_001",
    ...     image_url="https://example.com/image.jpg",
    ...     text_content="Caption text"
    ... )
    >>>
    >>> print(f"Consistency score: {result.consistency_score:.2%}")
    >>> print(f"Spurious correlation detected: {result.spurious_correlation_detected}")
"""

from .protected_attribute_schema import (
    ProtectedAttributeType,
    ModificationStrategy,
    ImageGenerationTool,
    ProtectedAttribute,
    CounterfactualVariation,
    ProtectedAttributeDetectionRequest,
    ProtectedAttributeDetectionResponse,
    CounterfactualGenerationRequest,
    CounterfactualGenerationResponse,
    CounterfactualTestCase,
    get_standard_variations,
    create_counterfactual_variation,
)

from .protected_attribute_detector import (
    ProtectedAttributeDetector,
    create_detector_from_config,
)

from .counterfactual_generator import (
    ImageGenerationAPI,
    DallE3API,
    StableDiffusionAPI,
    CounterfactualGenerator,
    create_generator_from_config,
    generate_all_variations,
)

from .automated_counterfactual_testing import (
    ConsistencyMetrics,
    AutomatedTestingConfig,
    AutomatedCounterfactualTester,
    create_tester_from_config,
)


__all__ = [
    # Schema
    "ProtectedAttributeType",
    "ModificationStrategy",
    "ImageGenerationTool",
    "ProtectedAttribute",
    "CounterfactualVariation",
    "ProtectedAttributeDetectionRequest",
    "ProtectedAttributeDetectionResponse",
    "CounterfactualGenerationRequest",
    "CounterfactualGenerationResponse",
    "CounterfactualTestCase",
    "get_standard_variations",
    "create_counterfactual_variation",
    # Detector
    "ProtectedAttributeDetector",
    "create_detector_from_config",
    # Generator
    "ImageGenerationAPI",
    "DallE3API",
    "StableDiffusionAPI",
    "CounterfactualGenerator",
    "create_generator_from_config",
    "generate_all_variations",
    # Testing
    "ConsistencyMetrics",
    "AutomatedTestingConfig",
    "AutomatedCounterfactualTester",
    "create_tester_from_config",
]
