import requests
import uuid
from typing import Optional, Dict, Any

class SovereignBrain:
    def __init__(self, base_url: str = "http://localhost:8001", api_key: str = "", default_model: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key  # <--- O PASSAPORTE
        }
        
        # Health Check silencioso
        try:
            requests.get(f"{self.base_url}/docs", headers=self.headers, timeout=2)
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
            # Header injetado aqui
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return "[ERRO 403] Acesso Negado: Verifique sua API Key."
            return f"[ERRO API] {e}"
        except Exception as e:
            return f"[ERRO SDK] {str(e)}"

    def learn(self, fact: str, session_id: str = "default") -> str:
        return self._send_payload(session_id, fact, model="memory-only")

    def ask(self, question: str, session_id: str = "default", model: Optional[str] = None) -> str:
        target_model = model or self.default_model
        return self._send_payload(session_id, question, model=target_model)

    # --- CRUD METHODS (Novos) ---
    
    def list_recent(self, limit: int = 10) -> list:
        try:
            url = f"{self.base_url}/v1/memories?limit={limit}"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 403: return ["Erro: Chave inválida"]
            return resp.json().get("data", [])
        except Exception as e:
            return [f"Erro: {str(e)}"]

    def forget(self, memory_id: int) -> str:
        try:
            url = f"{self.base_url}/v1/memories/{memory_id}"
            resp = requests.delete(url, headers=self.headers, timeout=5)
            return "Memória apagada." if resp.status_code == 200 else f"Erro: {resp.text}"
        except Exception as e:
            return f"Erro: {str(e)}"

    def update(self, memory_id: int, new_text: str) -> str:
        try:
            url = f"{self.base_url}/v1/memories/{memory_id}"
            resp = requests.put(url, json={"content": new_text}, headers=self.headers, timeout=5)
            return "Memória atualizada." if resp.status_code == 200 else f"Erro: {resp.text}"
        except Exception as e:
            return f"Erro: {str(e)}"
