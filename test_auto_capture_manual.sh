#!/bin/bash
# Test script for Auto-Capture feature
# Usage: ./test_auto_capture_manual.sh

set -e

API_KEY="sk_aethera_root_a15d11fb7aeefd16b16e96f33e06a9b0"
BASE_URL="http://localhost:8001"

echo "======================================"
echo "  Auto-Capture - Teste Manual"
echo "======================================"
echo ""

# 1. Enable auto-capture
echo "1️⃣  Ativando auto-capture..."
curl -s -X POST "$BASE_URL/v1/auto-capture/enable" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-demo"}' | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 2. Check status
echo "2️⃣  Verificando status..."
curl -s "$BASE_URL/v1/auto-capture/status?session_id=test-demo" \
  -H "x-api-key: $API_KEY" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 3. Capture command event
echo "3️⃣  Capturando evento de comando..."
curl -s -X POST "$BASE_URL/v1/auto-capture/event" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-demo",
    "event_type": "command",
    "event_data": {
      "command": "git commit -m \"feat: implement auto-capture feature\"",
      "exit_code": 0
    }
  }' | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 4. Capture file edit event
echo "4️⃣  Capturando evento de file edit..."
curl -s -X POST "$BASE_URL/v1/auto-capture/event" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-demo",
    "event_type": "file_edit",
    "event_data": {
      "file_path": "app/auto_capture.py",
      "lines_added": 300,
      "lines_removed": 0,
      "description": "Created auto-capture engine with detectors"
    }
  }' | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 5. Capture error event
echo "5️⃣  Capturando evento de erro..."
curl -s -X POST "$BASE_URL/v1/auto-capture/event" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-demo",
    "event_type": "error",
    "event_data": {
      "message": "Error: Connection timeout on PostgreSQL",
      "solution": "Increased timeout from 5s to 30s in connection pool",
      "context": "During background worker event processing"
    }
  }' | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 6. Capture decision event
echo "6️⃣  Capturando evento de decisão..."
curl -s -X POST "$BASE_URL/v1/auto-capture/event" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-demo",
    "event_type": "decision",
    "event_data": {
      "message": "Decidimos usar schedule library para background worker em vez de celery",
      "rationale": "Mais simples, menos dependências, suficiente para processar eventos a cada 30s",
      "alternatives": ["celery + redis", "APScheduler", "custom threading"]
    }
  }' | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 7. Check status again
echo "7️⃣  Status após captura de eventos..."
curl -s "$BASE_URL/v1/auto-capture/status?session_id=test-demo" \
  -H "x-api-key: $API_KEY" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 8. Wait for background processing
echo "⏰ Aguardando 30 segundos para o background worker processar..."
sleep 30
echo ""

# 9. Check status after processing
echo "8️⃣  Status após processamento..."
curl -s "$BASE_URL/v1/auto-capture/status?session_id=test-demo" \
  -H "x-api-key: $API_KEY" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))"
echo ""

# 10. Check memories
echo "9️⃣  Memórias criadas (últimas 10)..."
curl -s "$BASE_URL/v1/memories?limit=10" \
  -H "x-api-key: $API_KEY" | python3 -c "import sys, json; data = json.load(sys.stdin); [print(f\"  - [{m['id']}] {m['content'][:100]}...\") for m in data['data'][:10]]"
echo ""

# 11. Check usage stats
echo "🔟 Usage stats (deve incluir auto_capture)..."
curl -s "$BASE_URL/v1/usage" \
  -H "x-api-key: $API_KEY" | python3 -c "import sys, json; data = json.load(sys.stdin); print(json.dumps(data['usage']['auto_capture'], indent=2))"
echo ""

echo "======================================"
echo "✅ Testes completos!"
echo "======================================"
