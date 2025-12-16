"""
Metrics collection for synthesis operations.

This module provides dataclasses and utilities for tracking performance
and operational metrics during synthesis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class SynthesisMetrics:
    """Metrics collected during a synthesis operation.

    Attributes
    ----------
    agent_response_times : List[float]
        Response time in seconds for each agent
    total_retries : int
        Total number of retries across all agents
    validation_failures : int
        Number of validation failures before success
    aggregation_time : float
        Time taken for aggregation in seconds
    total_time : float
        Total synthesis time in seconds
    timestamp : datetime
        Timestamp when synthesis started
    agent_names : List[str]
        Names of agents that participated
    options_count : int
        Number of options presented
    method : str
        Aggregation method used
    timeout_count : int
        Number of timeout errors encountered
    errors : List[Dict[str, Any]]
        List of errors encountered during synthesis
    """

    agent_response_times: List[float] = field(default_factory=list)
    total_retries: int = 0
    validation_failures: int = 0
    aggregation_time: float = 0.0
    total_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_names: List[str] = field(default_factory=list)
    options_count: int = 0
    method: str = "majority"
    timeout_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'agent_response_times': self.agent_response_times,
            'avg_response_time': (
                sum(self.agent_response_times) / len(self.agent_response_times)
                if self.agent_response_times else 0.0
            ),
            'total_retries': self.total_retries,
            'validation_failures': self.validation_failures,
            'aggregation_time': self.aggregation_time,
            'total_time': self.total_time,
            'timestamp': self.timestamp.isoformat(),
            'agent_names': self.agent_names,
            'agent_count': len(self.agent_names),
            'options_count': self.options_count,
            'method': self.method,
            'timeout_count': self.timeout_count,
            'error_count': len(self.errors),
            'errors': self.errors,
        }

    def add_agent_time(self, agent_name: str, response_time: float):
        """Record response time for an agent."""
        self.agent_names.append(agent_name)
        self.agent_response_times.append(response_time)

    def add_error(self, error_type: str, message: str, agent: Optional[str] = None):
        """Record an error that occurred during synthesis."""
        self.errors.append({
            'type': error_type,
            'message': message,
            'agent': agent,
            'timestamp': datetime.utcnow().isoformat()
        })

    def increment_retries(self, count: int = 1):
        """Increment retry counter."""
        self.total_retries += count

    def increment_validation_failures(self, count: int = 1):
        """Increment validation failure counter."""
        self.validation_failures += count

    def increment_timeouts(self, count: int = 1):
        """Increment timeout counter."""
        self.timeout_count += count
