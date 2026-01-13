# Use uma imagem base leve e oficial
FROM python:3.10-slim

# Variáveis de ambiente para evitar arquivos .pyc e logs bufferizados
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Diretório de trabalho
WORKDIR /app

# Instalação de dependências do sistema necessárias para compilação
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY ./app /app/app

# Cria o diretório de dados e ajusta permissões
RUN mkdir -p /app/data

# Expõe a porta do Gateway de Memória
EXPOSE 8001

# Comando de execução
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
