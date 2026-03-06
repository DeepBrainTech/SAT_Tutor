"""
LLM Client Base Class
Defines a unified interface for different implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """LLM Response"""
    content: str
    success: bool
    error: Optional[str] = None
    raw_response: Optional[dict] = None


class LLMClient(ABC):
    """Abstract Base Class for LLM Client"""
    
    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Optional[str] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.1
    ) -> LLMResponse:
        """
        Generates a response in JSON format
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            schema_hint: JSON schema hint
            images: List of image paths (for vision models)
            temperature: Temperature parameter (0.0-1.0)
        
        Returns:
            LLMResponse containing the response content
        """
        pass
    
    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> LLMResponse:
        """
        Generates a plain text response
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature parameter
        
        Returns:
            LLMResponse containing the response content
        """
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the client is available"""
        pass