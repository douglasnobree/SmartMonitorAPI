#!/bin/bash

# Aguardar o banco de dados estar disponível
echo "Aplicando migrações..."
python manage.py migrate --no-input

# Criar superusuário se não existir
echo "Criando superusuário..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='${DJANGO_SUPERUSER_USERNAME:-admin}').exists():
    User.objects.create_superuser(
        username='${DJANGO_SUPERUSER_USERNAME:-admin}',
        email='${DJANGO_SUPERUSER_EMAIL:-admin@example.com}',
        password='${DJANGO_SUPERUSER_PASSWORD:-admin123}'
    )
    print('Superusuário criado com sucesso!')
else:
    print('Superusuário já existe!')
EOF

# Coletar arquivos estáticos
echo "Coletando arquivos estáticos..."
python manage.py collectstatic --no-input

# Iniciar servidor Gunicorn
echo "Iniciando servidor Gunicorn..."
exec gunicorn projectSM.wsgi:application --bind 0.0.0.0:8000 --workers 3
