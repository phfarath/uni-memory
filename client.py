import requests
import uuid
from typing import Optional, Dict, Any

class SovereignBrain:
    def __init__(self, base_url: str = "http://localhost:8001", api_key: str = "minha-senha-super-secreta-dev", default_model: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key 
        }

        # Verifica conexão na inicialização
        try:
            # Note: health check endpoint might not be auth protected, but let's assume it exists or fail gracefully
             requests.get(f"{self.base_url}/health", timeout=2)
        except Exception:
            pass 

    def _send_payload(self, session_id: str, prompt: str, model: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "session_id": session_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return "[ERRO FATAL] Acesso Negado: Chave de API inválida."
            return f"[ERRO API] Status {e.response.status_code}: {e.response.text}"
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

    def list_recent(self, limit: int = 10) -> list:
        """Retorna a lista crua de memórias com seus IDs."""
        try:
            url = f"{self.base_url}/v1/memories?limit={limit}"
            resp = requests.get(url, headers=self.headers, timeout=5)
            # The endpoint returns {"data": [...]}, so we access .get("data")
            return resp.json().get("data", [])
        except Exception as e:
            # Returning a list with error string to be compatible with expected list return, 
            # though caller should handle types. Logic follows user snippet.
            return [{"error": f"Erro ao listar: {str(e)}"}]

    def forget(self, memory_id: int) -> str:
        """Apaga uma memória específica pelo ID."""
        try:
            url = f"{self.base_url}/v1/memories/{memory_id}"
            resp = requests.delete(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                return f"Memória {memory_id} apagada com sucesso."
            return f"Erro ao apagar: {resp.text}"
        except Exception as e:
            return f"Erro de conexão: {str(e)}"

    def update(self, memory_id: int, new_text: str) -> str:
        """Reescreve uma memória existente."""
        try:
            url = f"{self.base_url}/v1/memories/{memory_id}"
            resp = requests.put(url, json={"content": new_text}, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                return f"Memória {memory_id} atualizada."
            return f"Erro ao atualizar: {resp.text}"
        except Exception as e:
            return f"Erro de conexão: {str(e)}"
