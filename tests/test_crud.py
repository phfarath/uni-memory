from client import SovereignBrain

brain = SovereignBrain(base_url="http://localhost:8001")

print(">>> [TESTE] Inserindo memória temporária...")
res = brain.learn("Esta é uma memória temporária para teste de CRUD.", session_id="test-crud")
print(f"Inserido: {res}")

print("\n>>> [TESTE] Listando memórias recentes...")
memories = brain.list_recent(limit=3)
target_id = None
for m in memories:
    print(f"ID: {m['id']} | Content: {m['content']}")
    if "memória temporária" in m['content']:
        target_id = m['id']

if target_id:
    print(f"\n>>> [TESTE] Atualizando memória ID {target_id}...")
    print(brain.update(target_id, "Memória temporária ATUALIZADA."))
    
    # Valida atualização
    updated_memories = brain.list_recent(limit=3)
    for m in updated_memories:
        if m['id'] == target_id:
            print(f"Verificação Pós-Update: {m['content']}")

    print(f"\n>>> [TESTE] Deletando memória ID {target_id}...")
    print(brain.forget(target_id))
    
    # Valida deleção
    final_memories = brain.list_recent(limit=3)
    found = any(m['id'] == target_id for m in final_memories)
    if not found:
        print("CONFIRMADO: ID não existe mais na lista.")
    else:
        print("FALHA: ID ainda persiste.")
else:
    print("ERRO: Não encontrei a memória inserida para testar delete.")
