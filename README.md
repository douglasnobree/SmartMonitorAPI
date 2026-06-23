# 🌊 SmartMonitor API

API Django REST para análise de dados de consumo de água utilizando Machine Learning e análise estatística.

## 📋 Descrição

Esta API fornece serviços de análise e predição de consumo de água através de endpoints REST. Ela é consumida por uma API backend principal que exibe os resultados no frontend.

### Funcionalidades Principais

- **Predição de Consumo**: Predição diária e mensal usando Regressão Linear
- **Análise Estatística**: Classificação de consumo usando Bandas de Bollinger
- **Dados Estatísticos**: Fornece dados completos das bandas para visualização
- **Classificação de pH**: Classificação de qualidade da água usando modelos ML por cliente

## 🏗️ Arquitetura

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────┐
│   Frontend  │ ───> │  API Backend     │ ───> │ SmartMonitor │
│             │      │  (Principal)     │      │     API      │
└─────────────┘      └──────────────────┘      └──────────────┘
                                                       │
                                                       ▼
                                                 ML Pipeline
                                            (Análise + Predição)
```

## 🚀 Tecnologias

- **Python 3.11**
- **Django 5.1.4**
- **Django REST Framework 3.15.2**
- **drf-yasg** (Swagger/OpenAPI)
- **JWT Authentication**
- **Pandas & NumPy** (Processamento de dados)
- **Scikit-learn** (Machine Learning)
- **Docker & Docker Compose**
- **Gunicorn** (WSGI Server)

## 📦 Instalação

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (opcional)

### Instalação Local

1. **Clone o repositório**
```bash
git clone https://github.com/douglasnobree/SmartMonitorAPI.git
cd SmartMonitorAPI
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Configure as variáveis de ambiente**
```bash
# Copie o arquivo de exemplo
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows

# Edite o .env e configure sua SECRET_KEY
# Gere uma nova chave com:
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

4. **Instale as dependências**
```bash
pip install -r requirements.txt
```

5. **Execute as migrações**
```bash
python manage.py migrate
```

6. **Crie um superusuário**
```bash
python manage.py createsuperuser
```

7. **Inicie o servidor**
```bash
python manage.py runserver
```

### Instalação com Docker

1. **Build e execute os containers**
```bash
docker-compose up --build
```

2. **Acesse a aplicação**
- API: http://localhost:8000
- Documentação: http://localhost:8000/ (Swagger), http://localhost:8000/swagger e http://localhost:8000/redoc

## 📚 Documentação da API

A documentação interativa está disponível via Swagger UI:

**URL:** `http://localhost:8000/`

### Autenticação

Os endpoints de negocio (`/prediction/*`, `/statistic/*`, `/classify/ph` e `/v2/*`) requerem autenticação JWT.
Os endpoints públicos são `/token`, `/`, `/swagger` e `/redoc`.

#### Obter Token

```bash
POST /token
Content-Type: application/json

{
    "username": "seu_usuario",
    "password": "sua_senha"
}
```

#### Usar Token

```bash
Authorization: seu_token_jwt
```

Observação: o token é enviado sem prefixo `Bearer`.

### Endpoints Principais

#### 🔮 Predição

- `POST /prediction/daily` - Predição de consumo diário
- `POST /prediction/monthly` - Predição de consumo mensal

**Formato de entrada:**
```json
{
    "01/06/2025": 120.5,
    "02/06/2025": 115.2,
    "03/06/2025": 130.0
}
```

#### 📊 Análise Estatística

- `POST /statistic/daily` - Classificação de consumo diário
- `POST /statistic/monthly` - Classificação de consumo mensal
- `POST /statistic/data` - Dados completos das bandas de Bollinger

**Formato de entrada:**
```json
{
    "01/06/2025": 120.5,
    "02/06/2025": 115.2,
    "03/06/2025": 130.0
}
```

#### 💧 Classificação de pH

- `POST /classify/ph` - Classificação de pH da água

Observação: o classificador de pH atual é um fluxo inicial de teste e, no estado presente, retorna uma classe simples de faixa/estado. O contrato pode evoluir no futuro.

**Formato de entrada:**
```json
{
    "client_id": "sisar",
    "ph_value": 7.2
}
```

#### Versão 2

A versão `v2` continua autenticada, mas busca o histórico em banco externo somente leitura.
Essa integração é pensada para estabilidade de longo prazo, mas depende de um banco mantido por terceiros.

- `POST /v2/prediction/daily` - Predição diária por `sensor_id`
- `POST /v2/prediction/monthly` - Predição mensal por `unidade_id` e, opcionalmente, `dispositivo_id`
- `POST /v2/statistic/daily` - Classificação diária por `sensor_id`
- `POST /v2/statistic/monthly` - Classificação mensal por `unidade_id` e, opcionalmente, `dispositivo_id`
- `POST /v2/statistic/data` - Dados completos das bandas diárias por `sensor_id`

**Exemplos de entrada v2:**
```json
{
    "sensor_id": "SENSOR-001"
}
```

```json
{
    "unidade_id": 12,
    "dispositivo_id": "disp-123"
}
```

#### Rotas de infraestrutura

- `GET /` - Swagger UI
- `GET /swagger` - Swagger UI
- `GET /redoc` - Redoc UI
- `GET /admin` - Django Admin
- `POST /token` - Obtenção de access/refresh JWT

## 🧪 Como Funciona

### Pipeline de Predição

1. API recebe dados históricos de consumo
2. Modelo de Regressão Linear é treinado com os dados
3. Predição é realizada
4. Resultado é retornado na mesma requisição

**Nota:** O modelo atual não é persistido. Cada requisição treina um novo modelo.

### Análise Estatística (Bandas de Bollinger)

1. Calcula média móvel (janela de 30 dias)
2. Calcula desvio padrão
3. Define bandas superior e inferior
4. Classifica o consumo em 5 categorias:
   - **Faixa inferior 2** (muito abaixo)
   - **Faixa inferior 1** (abaixo)
   - **Faixa ideal** (normal)
   - **Faixa superior 1** (acima)
   - **Faixa superior 2** (muito acima)

### Observações operacionais

- O endpoint `/token` retorna `access` e `refresh`.
- A rota de refresh dedicada está comentada no código atual.
- A ordenação do histórico é feita por data após normalização, e datas duplicadas são agregadas pela mediana.

## 📁 Estrutura do Projeto

```
SmartMonitorAPI/
├── appSM/                      # App Django principal
│   ├── views.py               # Views da API
│   ├── serializers.py         # Serializers DRF
│   └── ...
├── projectSM/                  # Configurações Django
│   ├── settings.py            # Configurações
│   ├── urls.py                # Rotas
│   └── authentication.py      # JWT customizado
├── ml_pipeline/               # Pipeline de Machine Learning
│   ├── Tratamento.py          # Interface abstrata
│   ├── senseFlow_A/           # Análise de consumo de água
│   │   ├── classificacao/     # Serviços de classificação
│   │   ├── modelos/           # Modelos ML
│   │   └── predicao/          # Serviços de predição
│   ├── senseflowQ/            # Análise de qualidade de água
│   │   └── ph_classification/ # Classificação de pH
│   └── models/                # Modelos ML armazenados
│       └── ph_classification/ # Modelos por cliente
├── static/                    # Arquivos estáticos
├── staticfiles/               # Arquivos coletados
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Imagem Docker
├── docker-compose.yml         # Orquestração Docker
└── manage.py                  # CLI Django
```

## 🔄 Roadmap Futuro

- [ ] **Análise de Qualidade de Água**: Expansão para métricas adicionais de qualidade
- [ ] **Testes Automatizados**: Cobertura completa de testes unitários e integração
- [ ] **Métricas de Performance**: Dashboard de monitoramento de API
- [ ] **Alertas Automáticos**: Notificações para consumo anômalo

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença BSD.

## 👥 Equipe

- **Contato:** RESOURCIFYLTDA@GMAIL.COM
- **Site:** https://www.senseflow.com.br/

## 📝 Notas de Desenvolvimento

### Ambiente de Desenvolvimento

O projeto usa PM2 para gerenciamento de processos em produção:

```bash
pm2 start ecosystem.config.js --env production
```

### Coleta de Arquivos Estáticos

```bash
python manage.py collectstatic --no-input
```

### Migrações

```bash
python manage.py makemigrations
python manage.py migrate
```

---

**Desenvolvido com ❤️ pela equipe SmartMonitor - IFCE**
