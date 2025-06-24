FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configurações padrão para o superusuário
ENV DJANGO_SUPERUSER_USERNAME=admin
ENV DJANGO_SUPERUSER_EMAIL=admin@example.com
ENV DJANGO_SUPERUSER_PASSWORD=admin123

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Criar diretório static
RUN mkdir -p static

# Tornar o script de inicialização executável
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

# Usar o script de inicialização
CMD ["./entrypoint.sh"]