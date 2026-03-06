"""
OpenAI API Client Implementation
Supports GPT-4 Vision and Text models
"""

import os
import base64
from typing import Optional

from .base import LLMClient, LLMResponse
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient(LLMClient):
    """
    OpenAI-compatible API Client
    
    Supports OpenAI, DeepSeek, Zhipu, Moonshot, and other APIs 
    compatible with the OpenAI API format.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the API client
        
        Args:
            api_key: API Key, defaults to reading from environment variables
            vision_model: Name of the vision model
            text_model: Name of the text model
            api_base: API Base URL, supports third-party compatible interfaces
                      Example: https://api.deepseek.com
                               https://api.moonshot.cn/v1
                               http://localhost:11434/v1 (Ollama)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")  # None defaults to OpenAI
        self.vision_model = vision_model or os.getenv("OPENAI_MODEL_VISION", "gpt-4o")
        self.text_model = text_model or os.getenv("OPENAI_MODEL_TEXT", "gpt-4o-mini")
        
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                # Support custom base_url
                if self.api_base:
                    self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                else:
                    self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    @property
    def is_available(self) -> bool:
        """Check if the client is available"""
        return bool(self.api_key and self._client)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Get image MIME type"""
        ext = os.path.splitext(image_path)[1].lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/png")
    
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Optional[str] = None,
        images: Optional[list[str]] = None,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generate a response in JSON format"""
        if not self.is_available:
            return LLMResponse(
                content="",
                success=False,
                error="OpenAI client unavailable, please check your API Key"
            )
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Build user message content
        user_content = []
        
        # Add images if provided
        if images:
            for image_path in images:
                if os.path.exists(image_path):
                    base64_image = self._encode_image(image_path)
                    media_type = self._get_image_media_type(image_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    })
        
        # Add text prompt
        full_prompt = user_prompt
        if schema_hint:
            full_prompt += f"\n\nExpected JSON Schema:\n{schema_hint}"
        
        user_content.append({"type": "text", "text": full_prompt})
        messages.append({"role": "user", "content": user_content})
        
        # Select model
        model = self.vision_model if images else self.text_model
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=4096,
                response_format={"type": "json_object"} if not images else None
            )
            
            content = response.choices[0].message.content
            return LLMResponse(
                content=content,
                success=True,
                raw_response=response.model_dump()
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"API call failed: {str(e)}"
            )
    
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> LLMResponse:
        """Generate a plain text response"""
        if not self.is_available:
            return LLMResponse(
                content="",
                success=False,
                error="OpenAI client unavailable, please check your API Key"
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=4096
            )
            
            content = response.choices[0].message.content
            return LLMResponse(
                content=content,
                success=True,
                raw_response=response.model_dump()
            )
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                error=f"API call failed: {str(e)}"
            )