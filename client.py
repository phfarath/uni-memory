import requests
import uuid
from typing import Optional, Dict, Any

class SovereignBrain:
    def __init__(self, base_url: str = "http://localhost:8001", default_model: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        # Verifica conexão na inicialização
        try:
            requests.get(f"{self.base_url}/health", timeout=2) # Se tiver endpoint de health (opcional)
        except Exception:
            pass # Ignora silenciosamente ou loga warning

    def _send_payload(self, session_id: str, prompt: str, model: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "session_id": session_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.ConnectionError:
            return "[ERRO FATAL] O Cérebro (Docker) está desligado ou inacessível."
        except Exception as e:
            return f"[ERRO SDK] Falha na comunicação: {str(e)}"

    def learn(self, fact: str, session_id: str = "default") -> str:
        """
        Ensina um fato ao cérebro sem gastar tokens de LLM (Ingestão Pura).
        """
        return self._send_payload(session_id, fact, model="memory-only")

    def ask(self, question: str, session_id: str = "default", model: Optional[str] = None) -> str:
        """
        Faz uma pergunta ao LLM usando o contexto histórico recuperado.
        """
        target_model = model or self.default_model
        return self._send_payload(session_id, question, model=target_model)

    def new_session(self) -> str:
        """Gera um ID de sessão único."""
        return str(uuid.uuid4())
