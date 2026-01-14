import requests
import sys

# Configurações
TARGET_URL = "http://100.49.87.83:8001" # IP que você definiu no mcp_server.py
TEST_KEY = "sk_aethera_kJIhaMXBPt-DaHb9uPjWPQ"

def check_connectivity():
    print(f"1. Testando conexão com {TARGET_URL}...")
    try:
        requests.get(f"{TARGET_URL}/docs", timeout=5)
        print("   [OK] Servidor acessível.")
        return True
    except Exception as e:
        print(f"   [FALHA] Não foi possível conectar: {e}")
        return False

def check_auth():
    print(f"\n2. Verificando Chave: {TEST_KEY[:15]}...")
    headers = {"x-api-key": TEST_KEY}
    
    try:
        # Tenta listar memórias (rota leve)
        resp = requests.get(f"{TARGET_URL}/v1/memories?limit=1", headers=headers, timeout=10)
        
        print(f"   Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            print("   [SUCESSO] A chave é VÁLIDA e tem acesso.")
            print(f"   Dados retornados: {resp.json()}")
        elif resp.status_code == 403:
            print("   [ERRO 403] A chave existe no .env/código mas NÃO está no Banco de Dados deste servidor.")
            print("   Dica: Você pode estar conectando no servidor errado ou o banco foi resetado.")
        else:
            print(f"   [ERRO {resp.status_code}] Resposta inesperada: {resp.text}")
            
    except Exception as e:
        print(f"   [ERRO] Falha na requisição: {e}")

if __name__ == "__main__":
    if check_connectivity():
        check_auth()
