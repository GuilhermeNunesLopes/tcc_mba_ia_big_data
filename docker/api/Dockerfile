# Usando uma imagem leve
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala as dependências necessárias
# (Flask para a API e Redis para a conexão com o banco)
RUN pip install --no-cache-dir flask redis

# Copia o código da API para dentro do container
COPY app.py .

# Expõe a porta que o Flask usa
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]