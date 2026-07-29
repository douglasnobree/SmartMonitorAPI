# 🌊 SmartMonitor API

API Django REST para análise de dados de consumo de água utilizando Machine Learning e análise estatística.

## 📋 Descrição

Esta API fornece serviços de análise e predição de consumo de água através de endpoints REST. Ela é consumida por uma API backend principal que exibe os resultados no frontend.

### Funcionalidades Principais

- **Predição de Consumo**: Predição diária e mensal usando Regressão Linear.
- **Análise Estatística**: Classificação de consumo e alertas usando Bandas de Bollinger.
- **Classificação de pH**: Classificação de qualidade da água usando modelos ML por cliente.

## 🏗️ Arquitetura

O projeto adota uma arquitetura modular orientada a domínio, dividida em quatro camadas principais para separar as responsabilidades e garantir a integridade dos dados:

- **API**: Camada de apresentação que recebe requisições, valida *payloads* via DRF Serializers e retorna respostas HTTP.
- **Services**: Orquestra os fluxos, solicita dados à infraestrutura e coordena chamadas à camada de domínio.
- **Domain**: Onde reside toda a inteligência matemática da aplicação (Regressão Linear, Bandas de Bollinger, limpeza e normalização de Pandas).
- **Infrastructure**: Responsável pelo acesso *read-only* eficiente a bancos de dados externos.

Para detalhes completos sobre o fluxo interno, o pipeline de Machine Learning e a organização de diretórios, consulte nosso documento dedicado de [Detalhes de Arquitetura](docs/architecture.md).

## 🧠 Conceitos de Negócio

Para facilitar o consumo da API, é importante entender os resultados gerados pelas análises estatísticas da plataforma.

### Classificação de Consumo (Bandas de Bollinger)
As Bandas de Bollinger são utilizadas para criar limites dinâmicos superiores e inferiores no histórico de consumo do usuário, baseando-se na média móvel e desvio padrão. Com isso, a API classifica o comportamento de consumo em **5 faixas**:

- **`-2` (Muito abaixo do esperado)**: O consumo caiu drasticamente em relação ao comportamento padrão.
- **`-1` (Abaixo do esperado)**: O consumo reduziu levemente.
- **`0` (Faixa normal / Green range)**: O consumo está perfeitamente dentro da normalidade estatística esperada para o período e usuário.
- **`1` (Acima do esperado)**: O consumo aumentou consideravelmente, servindo de alerta.
- **`2` (Muito acima do esperado)**: Um pico extremo de consumo, fortíssimo indício de vazamento ou desperdício crônico.

**Importante:** A API considera que a faixa ideal/alvo é sempre a faixa verde (`0`). Valores negativos representam economia e valores positivos representam alertas de excesso.

## 🚀 Tecnologias

- **Python 3.11** / **Django 5.1.4** / **DRF 3.15.2**
- **drf-yasg** (Swagger/OpenAPI)
- **JWT Authentication**
- **Pandas & NumPy** (Processamento de dados)
- **Scikit-learn** (Machine Learning)
- **Docker & Gunicorn**

## 📦 Instalação

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (opcional)

### Instalação Local

1. Clone o repositório e crie um ambiente virtual:
```bash
git clone https://github.com/douglasnobree/SmartMonitorAPI.git
cd SmartMonitorAPI
python -m venv venv
source venv/bin/activate  # Linux/Mac (ou venv\Scripts\activate no Windows)
```

2. Configure o ambiente (edite o `.env` se necessário):
```bash
cp .env.example .env
```

3. Instale as dependências e rode o servidor:
```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
OBS. Para gerar arquivos estáticos do django, use:
```
python manage.py collectstatic
```

### Instalação com Docker

```bash
docker-compose up --build
```
A API estará rodando em http://localhost:8000.

## 📚 Documentação da API

A documentação interativa detalhada e ao vivo está disponível via Swagger UI em: `http://localhost:8000/`

### Autenticação (JWT)
Os endpoints de negócio requerem cabeçalho JWT:
`Authorization: seu_token_jwt` (observação: o token é enviado sem prefixo Bearer).

Para obter um token provisório de acesso, chame a rota pública `POST /token` com suas credenciais de usuário.

---

### Endpoints Principais da API (Exemplos)

#### 🔮 1. Predição Diária
Prediz o consumo futuro diário usando Regressão Linear acumulada.
**`POST /v2/prediction/daily`**

**Requisição:**
```json
{
    "sensor_id": "SENSOR-001"
}
```
**Resposta (200 OK):**
```json
{
    "Prediction": 1420.50
}
```

#### 📊 2. Estatística Mensal
Gera a previsão de faixa de Bollinger mensal.
**`POST /v2/statistic/monthly`**

**Requisição:**
```json
{
    "unidade_id": 12,
    "dispositivo_id": "disp-123"
}
```
**Resposta (200 OK):**
```json
{
    "Data": "2026-07-30",
    "Consumo": 15800.00,
    "classificacao": 1
}
```

#### 📅 3. Histórico de Faturamento
Calcula as classificações consolidadas para todas as datas de um recorte histórico.
**`POST /v2/classification/history`**

**Requisição:**
```json
{
    "type": "daily",
    "unidade_id": 12,
    "data_inicio": "2026-06-01",
    "data_fim": "2026-06-03"
}
```
**Resposta (200 OK):**
```json
{
    "results": [
        {
            "periodo": "2026-06-01",
            "consumo": 120.0,
            "classificacao": 0
        },
        {
            "periodo": "2026-06-02",
            "consumo": 125.0,
            "classificacao": 1
        }
    ]
}
```

#### 🚨 4. Alerta de Faixa (Ontem)
Informa de maneira prática se o consumo do dia anterior fugiu à normalidade (`-2`, `1` ou `2`).
**`POST /v2/classification/range`**

**Requisição:**
```json
{
    "unidade_id": 12
}
```
**Resposta (200 OK):**
```json
{
    "outside_green_range": true
}
```

#### 💧 5. Classificação de Qualidade (pH)
Classifica uma amostra de pH baseando-se em modelos locais (ML) pré-treinados individualmente por cliente.
**`POST /classify/ph`**

**Requisição:**
```json
{
    "client_id": "sisar",
    "ph_value": 7.2
}
```
**Resposta (200 OK):**
```json
{
    "client_id": "sisar",
    "ph_value": 7.2,
    "classification": "adequado",
    "confidence": 0.95,
    "model_version": "v1.0.0"
}
```

---

### Respostas de Erro

A API utiliza códigos HTTP padronizados para refletir falhas no consumo dos dados. Os erros nativos da aplicação mantêm a chave `error` na estrutura de resposta.

#### 400 Bad Request
Retornado caso o Payload da requisição falhe ao ser lido (ex: sintaxe inválida).
```json
{
    "error": "JSON mal formatado."
}
```

#### 401 Unauthorized
Retornado nas rotas protegidas quando o Token JWT está ausente, é inválido ou expirou.
```json
{
    "detail": "As credenciais de autenticação não foram fornecidas."
}
```

#### 404 Not Found
Ocorre caso o banco de dados principal de consumo não possua nenhum registro histórico para gerar o período ou a base solicitada.
```json
{
    "error": "Nenhum registro encontrado no periodo solicitado"
}
```

#### 422 Unprocessable Entity
Ocorre mediante erro na validação dos campos pelo DRF (ex: string passada onde pedia-se numérico, ou formato de data incorreto).
```json
{
    "error": "Parâmetros inválidos",
    "details": {
        "unidade_id": [
            "Um número inteiro válido é exigido."
        ]
    }
}
```

#### 500 Internal Server Error
Retornado no evento de uma falha genérica de processamento matricial ou de exceção não tratada na orquestração dos dados.
```json
{
    "error": "Erro interno."
}
```

## 🔄 Status de Validação

- [x] **Arquitetura Orientada a Camadas**: Migração para arquitetura modular adaptada.
- [x] **Suíte de Testes Automatizados 100% Verde**: Testes isolados garantindo a independência das camadas.
- [ ] **Métricas de Performance**: Dashboard de monitoramento de API.

## 🤝 Contribuindo

1. Faça o Fork
2. Crie sua Feature Branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Add NovaFeature'`)
4. Push para a Branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença BSD.

---

**Desenvolvido com ❤️ pela equipe SenseFlow - Resourcify**
