import requests
import os

BAD_KEY = "chave-errada-invasor"
GOOD_KEY = "sk_aethera_LCPxn6Bl46xPljDyksxQlw"
BASE_URL = "http://localhost:8001"

def test_access(key_name, key_value, expected_status):
    print(f"\n>>> [TESTE AUTH] Tentando acesso com {key_name}...")
    headers = {"x-api-key": key_value}
    try:
        # Tenta listar memórias (rota protegida)
        resp = requests.get(f"{BASE_URL}/v1/memories?limit=1", headers=headers, timeout=5)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == expected_status:
            print(" [PASS] Comportamento esperado.")
            return True
        else:
            print(f" [FAIL] Esperado {expected_status}, recebeu {resp.status_code}")
            return False
    except Exception as e:
        print(f" [ERRO] {str(e)}")
        return False

# 1. Teste de Invasão (Sem chave ou chave errada)
t1 = test_access("CHAVE RUIM", BAD_KEY, 403)

# 2. Teste Legítimo (Chave correta)
t2 = test_access("CHAVE MESTRA", GOOD_KEY, 200)

if t1 and t2:
    print("\n>>> SUCESSO: O sistema está blindado.")
else:
    print("\n>>> FALHA: A segurança está comprometida.")
