

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str | None = None

    def __str__(self) -> str:
        return f"LLMResponse(tokens: {self.input_tokens}→{self.output_tokens}, model: {self.model})"
    



class BaseLLMProvider(ABC):
   

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
       
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        pass
       

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
       

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
        

    def __str__(self) -> str:
        return f"{self.provider_name.upper()} Provider (model: {self.model})"


class ProviderError(Exception):
    pass



class APIConnectionError(ProviderError):
    pass
    



class RateLimitError(ProviderError):
    pass
   



class AuthenticationError(ProviderError):
    pass
    


class InvalidRequestError(ProviderError):
    pass
   

if __name__ == "__main__":
    print("Base LLM Provider Classes")
    print("=" * 60)
    print("\nYe file ek blueprint hai - abstract base class.")
    print("OpenAI aur Anthropic providers isko inherit karenge.\n")
    
    print("Required Methods:")
    print("  1. chat() - Send messages and get response")
    print("  2. count_tokens() - Count tokens in text")
    print("  3. provider_name - Provider ka naam\n")
    
    print("LLMResponse Structure:")
    response = LLMResponse(
        content="Hello! How can I help you?",
        input_tokens=10,
        output_tokens=7,
        model="gpt-4o-mini",
        finish_reason="stop"
    )
    print(f"  {response}")
    print(f"  Content: {response.content}")
    print(f"  Tokens: {response.input_tokens} → {response.output_tokens}")
    
    print("\n" + "=" * 60)

