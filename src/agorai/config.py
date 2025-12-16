"""
Configuration file for AgorAI package.

This file contains configurable parameters for the AgorAI package.
Each parameter includes a description and can be overridden by environment variables.
"""

import os
from typing import Optional


class Config:
    """Central configuration for AgorAI package."""

    # ============================================================================
    # TIMEOUT SETTINGS
    # ============================================================================

    # LLM Generation Timeout (seconds)
    # Maximum time to wait for an LLM response during generation.
    # If the LLM provider doesn't respond within this time, the request will timeout.
    # Default: 30 seconds
    # Override with: AGORAI_LLM_TIMEOUT environment variable
    LLM_TIMEOUT: float = float(os.getenv("AGORAI_LLM_TIMEOUT", "30.0"))

    # ============================================================================
    # RETRY SETTINGS
    # ============================================================================

    # Maximum Retries for Structured Responses
    # Number of retry attempts when an agent fails to provide properly formatted response.
    # Default: 2 retries (total 3 attempts)
    # Override with: AGORAI_MAX_RETRIES environment variable
    MAX_RETRIES: int = int(os.getenv("AGORAI_MAX_RETRIES", "2"))

    # ============================================================================
    # VALIDATION SETTINGS
    # ============================================================================

    # Maximum Prompt Length (characters)
    # Prevents excessively long prompts that could cause memory issues or token limit errors.
    # Approximately 100,000 characters ≈ 25,000 tokens for most models.
    # Default: 100,000 characters
    # Override with: AGORAI_MAX_PROMPT_LENGTH environment variable
    MAX_PROMPT_LENGTH: int = int(os.getenv("AGORAI_MAX_PROMPT_LENGTH", "100000"))

    # Minimum Number of Options
    # Minimum options required for structured synthesis.
    # Default: 2
    # Override with: AGORAI_MIN_OPTIONS environment variable
    MIN_OPTIONS: int = int(os.getenv("AGORAI_MIN_OPTIONS", "2"))

    # Maximum Number of Options
    # Maximum options allowed for structured synthesis.
    # Default: 20
    # Override with: AGORAI_MAX_OPTIONS environment variable
    MAX_OPTIONS: int = int(os.getenv("AGORAI_MAX_OPTIONS", "20"))

    # ============================================================================
    # TEMPERATURE BOUNDS
    # ============================================================================

    # Minimum Temperature
    # Lower bound for LLM temperature parameter.
    # Default: 0.0
    # Override with: AGORAI_MIN_TEMPERATURE environment variable
    MIN_TEMPERATURE: float = float(os.getenv("AGORAI_MIN_TEMPERATURE", "0.0"))

    # Maximum Temperature
    # Upper bound for LLM temperature parameter.
    # Default: 2.0
    # Override with: AGORAI_MAX_TEMPERATURE environment variable
    MAX_TEMPERATURE: float = float(os.getenv("AGORAI_MAX_TEMPERATURE", "2.0"))

    # ============================================================================
    # CIRCUIT BREAKER SETTINGS
    # ============================================================================

    # Circuit Breaker Failure Threshold
    # Number of consecutive failures before opening the circuit.
    # Default: 5 failures
    # Override with: AGORAI_CIRCUIT_BREAKER_FAILURES environment variable
    CIRCUIT_BREAKER_FAIL_MAX: int = int(os.getenv("AGORAI_CIRCUIT_BREAKER_FAILURES", "5"))

    # Circuit Breaker Reset Timeout (seconds)
    # Time to wait before attempting to close the circuit after it opens.
    # Default: 60 seconds
    # Override with: AGORAI_CIRCUIT_BREAKER_TIMEOUT environment variable
    CIRCUIT_BREAKER_RESET_TIMEOUT: int = int(os.getenv("AGORAI_CIRCUIT_BREAKER_TIMEOUT", "60"))

    # ============================================================================
    # LOGGING SETTINGS
    # ============================================================================

    # Log Directory
    # Directory where log files will be stored.
    # Default: ~/.agorai/logs (Unix/Mac) or %USERPROFILE%\.agorai\logs (Windows)
    # Override with: AGORAI_LOG_DIR environment variable
    @staticmethod
    def get_log_dir() -> str:
        """Get the logging directory path."""
        default_log_dir = os.path.join(
            os.path.expanduser("~"),
            ".agorai",
            "logs"
        )
        return os.getenv("AGORAI_LOG_DIR", default_log_dir)

    # Log Level
    # Logging level for the package.
    # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # Default: INFO
    # Override with: AGORAI_LOG_LEVEL environment variable
    LOG_LEVEL: str = os.getenv("AGORAI_LOG_LEVEL", "INFO")

    # Log File Max Size (bytes)
    # Maximum size of a single log file before rotation.
    # Default: 10 MB
    # Override with: AGORAI_LOG_MAX_BYTES environment variable
    LOG_MAX_BYTES: int = int(os.getenv("AGORAI_LOG_MAX_BYTES", "10485760"))  # 10 MB

    # Log File Backup Count
    # Number of backup log files to keep.
    # Default: 5
    # Override with: AGORAI_LOG_BACKUP_COUNT environment variable
    LOG_BACKUP_COUNT: int = int(os.getenv("AGORAI_LOG_BACKUP_COUNT", "5"))

    # ============================================================================
    # PERFORMANCE SETTINGS
    # ============================================================================

    # Enable Tokenization Caching
    # Whether to cache tokenization results for performance optimization.
    # Default: True
    # Override with: AGORAI_ENABLE_TOKEN_CACHE environment variable
    ENABLE_TOKEN_CACHE: bool = os.getenv("AGORAI_ENABLE_TOKEN_CACHE", "true").lower() == "true"


# Create a global config instance
config = Config()
