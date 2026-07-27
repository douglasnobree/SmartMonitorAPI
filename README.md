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

Os endpoints de negócio (`/v2/*` e `/classify/ph`) requerem autenticação JWT.
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

### Endpoints da API

#### 🔮 Predição e Análise Estatística (Versão 2)

A API v2 opera nativamente integrada ao banco de dados externo somente leitura de consumo, trazendo estabilidade e processamento em memória eficiente (sem conversões desnecessárias, atuando ponta a ponta com `pandas.DataFrame`).

- `POST /v2/prediction/daily` - Predição diária por `sensor_id`
- `POST /v2/prediction/monthly` - Predição mensal por `unidade_id` e, opcionalmente, `dispositivo_id`
- `POST /v2/statistic/daily` - Classificação diária de consumo por `sensor_id`
- `POST /v2/statistic/monthly` - Classificação mensal por `unidade_id` e, opcionalmente, `dispositivo_id`
- `POST /v2/statistic/data` - Dados completos das bandas diárias de Bollinger por `sensor_id`
- `POST /v2/classification/history` - Classificação histórica para relatórios por `unidade_id`, em modo `daily` ou `monthly`

**Exemplos de entrada V2:**
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

```json
{
    "type": "daily",
    "unidade_id": 12,
    "data_inicio": "2026-06-01",
    "data_fim": "2026-06-30"
}
```

#### 💧 Classificação de pH

- `POST /classify/ph` - Classificação de pH da água

Utiliza modelos locais personalizados treinados por cliente na camada de domínio.

**Formato de entrada:**
```json
{
    "client_id": "sisar",
    "ph_value": 7.2
}
```

#### Rotas de Infraestrutura

- `GET /` - Swagger UI
- `GET /swagger` - Swagger UI
- `GET /redoc` - Redoc UI
- `GET /admin` - Django Admin
- `POST /token` - Obtenção de access/refresh JWT

## 🧪 Como Funciona

### Pipeline de Predição e ML (Arquitetura Modulada)

1. A camada **API** valida a entrada de dados (via Serializers do DRF).
2. A camada de **Services** solicita o histórico em formato `DataFrame` diretamente à camada de **Infrastructure** (`db_fetcher.py` operando com *Module-level Lazy Engine* e conexão segura externa).
3. O pré-processamento e limpeza delegam para a camada de **Domain** via composição funcional (`tratamento.py`), tratando *gaps*, *outliers* de faturamento e aplicando medianas sem round-trips para dicionários na memória.
4. Os modelos matemáticos (Regressão Linear Acumulada e Bandas de Bollinger) efetuam as análises e predições instantâneas.

### Análise Estatística (Bandas de Bollinger)

1. Calcula média móvel (janela customizável/padrão de 30 dias).
2. Calcula desvio padrão da série.
3. Define bandas superior e inferior com fatiamento nativo em Pandas.
4. Classifica o consumo em 5 categorias consolidadas:
   - **Faixa inferior 2** (muito abaixo)
   - **Faixa inferior 1** (abaixo)
   - **Faixa ideal** (normal)
   - **Faixa superior 1** (acima)
   - **Faixa superior 2** (muito acima)

## 📁 Estrutura do Projeto (Clean Architecture)

A aplicação segue uma separação rigorosa de responsabilidades por camadas modulares:

```
SmartMonitorAPI/
├── appSM/                           # App Django principal modulado em 4 camadas:
│   ├── api/                         # 1. Camada de Apresentação e REST
│   │   ├── views.py                 # Endpoints DRF V2 e Classificação pH
│   │   └── serializers.py           # Contratos e validadores JSON
│   ├── services/                    # 2. Camada de Regras e Casos de Uso
│   │   ├── predicao_service.py      # Caso de uso: predição linear
│   │   ├── analise_estatistica_service.py # Caso de uso: bandas estatísticas
│   │   ├── classification_history_service.py # Caso de uso: séries de faturamento/relatórios
│   │   └── ph_classification_service.py      # Caso de uso: análise qualidade pH
│   ├── domain/                      # 3. Camada de Domínio, Algoritmos e Lógica
│   │   ├── tratamento.py            # Normalização, preenchimento de gaps e limpeza
│   │   ├── regressao_linear.py      # Modelagem matemática de regressão acumulada
│   │   └── models/                  # Diretório contendo persistência de modelos (.joblib)
│   ├── infrastructure/              # 4. Camada de Adaptadores Externos
│   │   └── db_fetcher.py            # Conexão com banco externo via Module-level Lazy Engine
│   ├── tests.py                     # Suíte de testes unitários da V2 e Serviços
│   └── test_characterization.py     # Suíte de testes de caracterização (regras ML invioláveis)
├── projectSM/                       # Configurações globais do projeto Django
│   ├── settings.py                  # Parâmetros gerais e diretórios de modelo
│   ├── urls.py                      # Roteamento central e Swagger
│   └── authentication.py            # Customização de JWT
├── static/                          # Arquivos estáticos
├── requirements.txt                 # Dependências Python (pandas, scikit-learn, django, drf)
├── Dockerfile                       # Container Docker da aplicação
└── manage.py                        # CLI Django
```

## 🔄 Status de Validação

- [x] **Arquitetura Orientada a Camadas**: Migração para Clean Architecture (`api`, `services`, `domain`, `infrastructure`).
- [x] **Suíte de Testes Automatizados 100% Verde**: Testes unitários da API V2 e de caracterização cobrendo regressão linear, preenchimento de mediana e agregação sem perdas e sem endpoints legados obsoletos.
- [ ] **Análise de Qualidade de Água**: Expansão para métricas adicionais de qualidade (além do pH).
- [ ] **Métricas de Performance**: Dashboard de monitoramento de API.

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
