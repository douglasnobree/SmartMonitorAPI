# 🌊 SmartMonitor API

API Django REST para análise de dados de consumo de água utilizando Machine Learning e análise estatística.

## 📋 Descrição

Esta API fornece serviços de análise e predição de consumo de água através de endpoints REST. Ela é consumida por uma API backend principal que exibe os resultados no frontend.

### Funcionalidades Principais

- **Predição de Consumo**: Predição diária e mensal usando Regressão Linear
- **Análise Estatística**: Classificação de consumo usando Bandas de Bollinger
- **Dados Estatísticos**: Fornece dados completos das bandas para visualização

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

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute as migrações**
```bash
python manage.py migrate
```

5. **Crie um superusuário**
```bash
python manage.py createsuperuser
```

6. **Inicie o servidor**
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
- Documentação: http://localhost:8000/ (Swagger)

## 📚 Documentação da API

A documentação interativa está disponível via Swagger UI:

**URL:** `http://localhost:8000/`

### Autenticação

Todos os endpoints (exceto `/token`) requerem autenticação JWT.

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

- `POST /statistic/diaria` - Classificação de consumo diário
- `POST /statistic/mensal` - Classificação de consumo mensal
- `POST /statistic/data` - Dados completos das bandas de Bollinger

**Formato de entrada:**
```json
{
    "01/06/2025": 120.5,
    "02/06/2025": 115.2,
    "03/06/2025": 130.0
}
```

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
│   └── senseFlow_A/           # Análise de consumo de água
│       ├── classificacao/     # Serviços de classificação
│       ├── modelos/           # Modelos ML
│       └── predicao/          # Serviços de predição
├── static/                    # Arquivos estáticos
├── staticfiles/               # Arquivos coletados
├── requirements.txt           # Dependências Python
├── Dockerfile                 # Imagem Docker
├── docker-compose.yml         # Orquestração Docker
└── manage.py                  # CLI Django
```

## 🔄 Roadmap Futuro

- [ ] **Serviço de Retreino**: Sistema automatizado de retreinamento de modelos
- [ ] **Repositório de Modelos**: Armazenamento de modelos na nuvem (Drive inicial)
- [ ] **Multi-tenant**: Suporte a múltiplos clientes com modelos específicos
- [ ] **SenseFlow-Q**: Análise de qualidade de água
- [ ] **Versionamento de Modelos**: Controle de versões dos modelos ML
- [ ] **Testes Automatizados**: Cobertura completa de testes

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
- **Site:** https://www.smartmonitor.ifce.edu.br/

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
