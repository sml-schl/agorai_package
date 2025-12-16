"""
Circuit breaker pattern for external API calls.

This module implements the circuit breaker pattern to prevent cascading failures
when external LLM APIs are experiencing issues.
"""

import time
from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta

from ..config import config
from ..logging_config import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for external API calls.

    The circuit breaker monitors failures and opens the circuit when
    the failure threshold is exceeded, preventing further calls until
    a timeout period has elapsed.

    Parameters
    ----------
    fail_max : int
        Number of consecutive failures before opening circuit
    reset_timeout : int
        Seconds to wait before attempting to close circuit
    name : str
        Name of the circuit breaker (for logging)

    Examples
    --------
    >>> breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
    >>> try:
    ...     result = breaker.call(risky_function, arg1, arg2)
    ... except CircuitBreakerError:
    ...     print("Circuit is open, service unavailable")
    """

    def __init__(
        self,
        fail_max: Optional[int] = None,
        reset_timeout: Optional[int] = None,
        name: str = "default"
    ):
        self.fail_max = fail_max or config.CIRCUIT_BREAKER_FAIL_MAX
        self.reset_timeout = reset_timeout or config.CIRCUIT_BREAKER_RESET_TIMEOUT
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitState.CLOSED

        logger.info(
            f"CircuitBreaker '{self.name}' initialized: "
            f"fail_max={self.fail_max}, reset_timeout={self.reset_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                logger.info(f"CircuitBreaker '{self.name}' transitioned to HALF_OPEN")

        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False

        time_since_failure = datetime.utcnow() - self._last_failure_time
        return time_since_failure >= timedelta(seconds=self.reset_timeout)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.

        Parameters
        ----------
        func : Callable
            Function to call
        *args
            Positional arguments for the function
        **kwargs
            Keyword arguments for the function

        Returns
        -------
        Any
            Result of the function call

        Raises
        ------
        CircuitBreakerError
            If circuit is open
        Exception
            Any exception raised by the function
        """
        if self.state == CircuitState.OPEN:
            logger.warning(
                f"CircuitBreaker '{self.name}' is OPEN, blocking call to {func.__name__}"
            )
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is open. "
                f"Service unavailable. Try again in {self.reset_timeout}s."
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"CircuitBreaker '{self.name}' closed after successful test")

        # Reset failure count and close circuit
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time = None

    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()

        logger.warning(
            f"CircuitBreaker '{self.name}' failure {self._failure_count}/{self.fail_max}"
        )

        if self._failure_count >= self.fail_max:
            self._state = CircuitState.OPEN
            logger.error(
                f"CircuitBreaker '{self.name}' opened after {self._failure_count} failures. "
                f"Will retry after {self.reset_timeout}s"
            )

    def reset(self):
        """Manually reset the circuit breaker."""
        logger.info(f"CircuitBreaker '{self.name}' manually reset")
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time = None

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
            f"failures={self._failure_count}/{self.fail_max})"
        )
