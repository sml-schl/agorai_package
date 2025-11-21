"""LLM provider integrations.

Supports OpenAI, Anthropic, Ollama, and Google with a unified interface.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM.

        Returns
        -------
        Dict[str, Any]
            {
                'text': str,
                'scores': Dict[str, float],  # Optional
                'metadata': Dict[str, Any]
            }
        """
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference.

    Parameters
    ----------
    model : str
        Model name (e.g., "llama3.2", "mistral", "qwen3-vl:2b")
    base_url : str
        Ollama server URL (default: "http://localhost:11434")
    temperature : float
        Sampling temperature (default: 0.7)
    timeout : int
        Request timeout in seconds (default: 120)
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 120
    ):
        super().__init__(model, temperature)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from Ollama."""
        import requests

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            # Parse Ollama response
            data = response.json()
            text = data.get("response", "")

            return {
                'text': text,
                'scores': {},
                'metadata': {
                    'model': self.model,
                    'provider': 'ollama',
                    'base_url': self.base_url
                }
            }
        except Exception as e:
            return {
                'text': f"[ERROR] {str(e)}",
                'scores': {},
                'metadata': {
                    'model': self.model,
                    'provider': 'ollama',
                    'error': str(e)
                }
            }


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using official OpenAI SDK.

    Parameters
    ----------
    model : str
        Model name (e.g., "gpt-4", "gpt-4o", "gpt-3.5-turbo")
    api_key : str
        OpenAI API key
    temperature : float
        Sampling temperature (default: 0.7)
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.7):
        super().__init__(model, temperature)
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from OpenAI."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            text = response.choices[0].message.content

            return {
                'text': text,
                'scores': {},
                'metadata': {
                    'model': self.model,
                    'provider': 'openai',
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
            }
        except ImportError:
            return {
                'text': "[ERROR] OpenAI SDK not installed. Install with: pip install openai",
                'scores': {},
                'metadata': {'provider': 'openai', 'error': 'missing_sdk'}
            }
        except Exception as e:
            return {
                'text': f"[ERROR] {str(e)}",
                'scores': {},
                'metadata': {'provider': 'openai', 'error': str(e)}
            }


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider using official Anthropic SDK.

    Parameters
    ----------
    model : str
        Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
    api_key : str
        Anthropic API key
    temperature : float
        Sampling temperature (default: 0.7)
    max_tokens : int
        Maximum tokens to generate (default: 1024)
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        super().__init__(model, temperature)
        self.api_key = api_key
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from Anthropic."""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text

            return {
                'text': text,
                'scores': {},
                'metadata': {
                    'model': self.model,
                    'provider': 'anthropic',
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    }
                }
            }
        except ImportError:
            return {
                'text': "[ERROR] Anthropic SDK not installed. Install with: pip install anthropic",
                'scores': {},
                'metadata': {'provider': 'anthropic', 'error': 'missing_sdk'}
            }
        except Exception as e:
            return {
                'text': f"[ERROR] {str(e)}",
                'scores': {},
                'metadata': {'provider': 'anthropic', 'error': str(e)}
            }


class GoogleProvider(BaseLLMProvider):
    """Google Generative AI provider (Gemini).

    Parameters
    ----------
    model : str
        Model name (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
    api_key : str
        Google API key
    temperature : float
        Sampling temperature (default: 0.7)
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.7):
        super().__init__(model, temperature)
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from Google Gemini."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )

            text = response.text

            return {
                'text': text,
                'scores': {},
                'metadata': {
                    'model': self.model,
                    'provider': 'google'
                }
            }
        except ImportError:
            return {
                'text': "[ERROR] Google Generative AI SDK not installed. Install with: pip install google-generativeai",
                'scores': {},
                'metadata': {'provider': 'google', 'error': 'missing_sdk'}
            }
        except Exception as e:
            return {
                'text': f"[ERROR] {str(e)}",
                'scores': {},
                'metadata': {'provider': 'google', 'error': str(e)}
            }


def get_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseLLMProvider:
    """Factory function to get LLM provider instance.

    Parameters
    ----------
    provider : str
        Provider name ("openai", "anthropic", "ollama", "google")
    model : str
        Model identifier
    api_key : Optional[str]
        API key (not needed for Ollama)
    base_url : Optional[str]
        Custom base URL (for Ollama)
    temperature : float
        Sampling temperature
    **kwargs
        Additional provider-specific parameters

    Returns
    -------
    BaseLLMProvider
        Provider instance

    Raises
    ------
    ValueError
        If provider is unknown or required parameters are missing
    """
    provider = provider.lower()

    if provider == "ollama":
        base_url = base_url or "http://localhost:11434"
        return OllamaProvider(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )

    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires api_key")
        return OpenAIProvider(
            model=model,
            api_key=api_key,
            temperature=temperature
        )

    elif provider == "anthropic":
        if not api_key:
            raise ValueError("Anthropic provider requires api_key")
        return AnthropicProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )

    elif provider == "google":
        if not api_key:
            raise ValueError("Google provider requires api_key")
        return GoogleProvider(
            model=model,
            api_key=api_key,
            temperature=temperature
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, anthropic, ollama, google"
        )


def list_providers() -> Dict[str, Any]:
    """List all available LLM providers and their models.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping provider names to their available models
    """
    return {
        "openai": {
            "models": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo"
            ],
            "requires_api_key": True,
            "description": "OpenAI GPT models"
        },
        "anthropic": {
            "models": [
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ],
            "requires_api_key": True,
            "description": "Anthropic Claude models"
        },
        "ollama": {
            "models": [
                "llama3.2",
                "llama3.1",
                "llama2",
                "mistral",
                "mixtral",
                "gemma2",
                "qwen3-vl:2b"
            ],
            "requires_api_key": False,
            "description": "Local Ollama models (requires Ollama server)"
        },
        "google": {
            "models": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro"
            ],
            "requires_api_key": True,
            "description": "Google Gemini models"
        }
    }
