"""
Mock provider que simula chamadas à API OpenAI.
Usa async.sleep para simular latência de rede.
"""

import asyncio
import time
from typing import Dict, Any
from datetime import datetime


class MockOpenAIProvider:
    """
    Provider mock que simula a API da OpenAI.
    Gera respostas determinísticas baseadas no prompt.
    """
    
    # Simula latência da API OpenAI em segundos
    API_LATENCY = 0.8
    
    # Respostas mockadas para prompts comuns
    MOCK_RESPONSES = {
        "default": "Esta é uma resposta gerada pelo mock provider. Em produção, isso viria da OpenAI API."
    }
    
    @staticmethod
    def _generate_response(prompt: str) -> str:
        """Gera uma resposta baseada no prompt."""
        # Em produção, isso seria uma chamada real à OpenAI
        return f"Resposta para: '{prompt[:50]}...'\n\n" + MockOpenAIProvider.MOCK_RESPONSES["default"]
    
    @staticmethod
    async def chat_completions(request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula uma chamada à /v1/chat/completions da OpenAI.
        
        Args:
            request_body: Corpo da request no formato OpenAI
            
        Returns:
            Response no formato OpenAI
        """
        # Simular latência de rede
        await asyncio.sleep(MockOpenAIProvider.API_LATENCY)
        
        # Extrair mensagens
        messages = request_body.get("messages", [])
        model = request_body.get("model", "gpt-4")
        
        # Extrair o último prompt (última mensagem do usuário)
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        # Gerar resposta
        response_text = MockOpenAIProvider._generate_response(prompt)
        
        # Construir response no formato OpenAI
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        
        return response
    
    @staticmethod
    async def stream_chat_completions(request_body: Dict[str, Any]):
        """
        Simula streaming de chat completions (opcional para futuro).
        """
        # Placeholder para implementação futura
        raise NotImplementedError("Streaming não implementado no mock MVP")
