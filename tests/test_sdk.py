from client import SovereignBrain
import time

import os
# Inicializa o cliente apontando para seu Docker com a chave correta
api_key = os.environ.get("SOVEREIGN_KEY", "sk_aethera_kJIhaMXBPt-DaHb9uPjWPQ")
brain = SovereignBrain(base_url="http://100.49.87.83:8001", api_key=api_key)

# Cria uma sessão única para não misturar com testes anteriores
session_id = f"agente-007-{int(time.time())}"
print(f"--- INICIANDO SESSÃO: {session_id} ---\n")

# 1. FASE DE APRENDIZADO (Rápida e Gratuita)
fatos = [
    "O alvo se encontra no Hotel Continental, quarto 303.",
    "A senha de acesso ao servidor é 'TANGO-DOWN'.",
    "O contato de extração é a Agente Carter."
]

print(">>> [FASE 1] Ingestão de Dados (Memory-Only)...")
for fato in fatos:
    resp = brain.learn(fato, session_id=session_id)
    print(f" [OK] Ensinado: '{fato[:30]}...' -> Resposta: {resp}")

print("\n------------------------------------------------\n")

# 2. FASE DE CONSULTA (Inteligente)
perguntas = [
    "Onde o alvo está escondido?",
    "Qual a senha e quem vai me tirar de lá?"
]

print(">>> [FASE 2] Consultando Inteligência (GPT/Claude)...")
for p in perguntas:
    print(f"\n[USER]: {p}")
    # Aqui o 'brain.ask' faz toda a mágica de RAG + Injection
    resposta = brain.ask(p, session_id=session_id)
    print(f"[BRAIN]: {resposta}")

print("\n--- FIM DA OPERAÇÃO ---")
