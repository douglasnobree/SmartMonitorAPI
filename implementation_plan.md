# 🗺️ Plano de Implementação — Refatoração Arquitetural SmartMonitor API
### Versão Final — Decisões Consolidadas

> Escopo: consolidação de todos os itens da análise de débito técnico + pontos levantados pelo time.
> pH e features relacionadas: **fora do escopo** (adiado, candidato a remoção futura).

---

## Decisões Fechadas

| # | Decisão | Escolha |
|---|---|---|
| 1 | Granularidade dos repositórios | **Separar por entidade** — `ConsumoRepository` + `DispositivoRepository` |
| 2 | `AnaliseEstatisticaService` | **Dissolver completamente** nos métodos do use case |
| 3 | Organização dos testes | **Pasta própria** — `tests/` na raiz de `appSM/` |
| 4 | `ClassificationRangeService` | Injeção de dependência via construtor (resolvida na Fase 3) |
| + | `EstrategiaBollinger` | **Protocol adicionado** em `domain/bollinger.py` (extensibilidade) |
| + | `ModelStore` | **Adicionado** em `infrastructure/` (pipelines ML futuros) |

---

## 📐 Diagnóstico Crítico (Referência)

### Os 6 problemas que este plano resolve

**P1 — Limites de camada vazam:** `_V2BaseView` importa diretamente de `infrastructure` (`ExternalDataFetcher`, `ExternalDataNotFoundError`, `ExternalDeviceNotFoundError`). View deveria ser agnóstica à infraestrutura.

**P2 — `AnaliseEstatisticaService` é pipeline monolítico, não serviço:** 85% do seu código é algoritmo puro de domínio (outliers, bandas, classificação) que vive no lugar errado.

**P3 — `ExternalDataFetcher` tem lógica de negócio infiltrada:** `_aggregate_monthly` e lógica de datas de ciclo são regras de negócio faturamento, não infraestrutura.

**P4 — Flag `is_monthly` e proxy `janela == 12`:** O tipo de análise (diária/mensal) não é cidadão de primeira classe — é inferido implicitamente em vários pontos.

**P5 — Código zumbi:** `MySerializer`, `dataframe_para_historico`, métodos `train()`/`prediction()`, constante `JANELA_DIARIA = 7`.

**P6 — Artefatos ML mal posicionados:** `domain/models/` armazena arquivos `.joblib` (artefatos físicos = infraestrutura) dentro do domínio.

---

## 🏗️ Estrutura Final de Camadas

```
appSM/
│
├── api/                                   # Apresentação — HTTP only
│   ├── views.py                           # Views thin: valida → chama use case → retorna HTTP
│   └── serializers.py                     # Contratos DRF de entrada
│
├── application/                           # Casos de Uso (renomeado de services/)
│   ├── __init__.py
│   ├── exceptions.py                      # NOVO: ConsumoNaoEncontrado, DispositivoNaoEncontrado
│   ├── estatistica_use_case.py            # DISSOLVE AnaliseEstatisticaService (métodos estáticos)
│   ├── predicao_use_case.py               # Renomeado de predicao_service.py
│   ├── historico_use_case.py              # Renomeado de classification_history_service.py
│   └── range_use_case.py                  # Renomeado de classification_range_service.py
│
├── domain/                                # Regras e algoritmos puros — zero I/O, zero Django
│   ├── __init__.py
│   ├── tratamento.py                      # Já existe — normalização e gaps
│   ├── outliers.py                        # NOVO: algoritmo IQR unificado e parametrizável
│   ├── bollinger.py                       # NOVO: Protocol EstrategiaBollinger + implementação padrão
│   ├── classificador.py                   # NOVO: regras de faixa (-2…+2) + CLASSIFICATION_METADATA
│   ├── ciclo_faturamento.py               # NOVO: agregar_por_ciclo_mensal + periodos_do_ano
│   └── regressao_linear.py                # Já existe — mantém (sem métodos legados)
│
├── infrastructure/                        # I/O externo — banco, disco, modelos
│   ├── __init__.py
│   ├── engine.py                          # NOVO: singleton da engine SQL (Lazy)
│   ├── exceptions.py                      # NOVO: DataNotFoundError, DeviceNotFoundError
│   ├── consumo_repository.py              # NOVO: queries de consumo (extraído de db_fetcher)
│   ├── dispositivo_repository.py          # NOVO: query de dia_fechamento (extraído de db_fetcher)
│   └── model_store.py                     # NOVO: carregamento/salvamento de artefatos ML (.joblib)
│
└── tests/                                 # REORGANIZADO — saiu de appSM/tests.py
    ├── __init__.py
    ├── test_api.py                        # Testes de integração HTTP (ex-PredictionAndAnalysisAPITests)
    ├── test_use_cases.py                  # Testes unitários dos use cases
    └── test_characterization.py           # Testes de caracterização — contratos invioláveis
```

**O que desaparece:**
- `appSM/services/` — dissolvido em `application/`
- `appSM/infrastructure/db_fetcher.py` — dividido em `engine.py` + `consumo_repository.py` + `dispositivo_repository.py`
- `appSM/domain/models/` — **movido para** `appSM/infrastructure/models/`
- `appSM/tests.py` e `appSM/test_characterization.py` — movidos para `appSM/tests/`

---

## 🚫 O que não muda

- Contratos públicos da API (URLs, payloads JSON) — **zero breaking change externo**
- Lógica matemática dos algoritmos — os Testes de Caracterização são o contrato inviolável
- Autenticação JWT (`projectSM/authentication.py`)
- Passagem nativa de `DataFrame` end-to-end (sem round-trips dict↔DataFrame)
- pH e tudo relacionado — fora do escopo

---

## 📋 Fases de Implementação

---

### FASE 1 — Consolidar o domínio

**Objetivo:** `domain/` contém toda lógica pura — algoritmos, regras, sem I/O. Nenhum service implementa algoritmo.

**Regra:** Cada item desta fase é **independente** dos outros e pode ser feito em qualquer ordem. Todos os testes devem continuar passando ao fim de cada sub-item.

---

#### 1.1 · `domain/outliers.py` — algoritmo IQR unificado

**Problema resolvido:** Duplicação do IQR entre `PredicaoService._tratar_outliers_mediana` (IQR 3.0×, mediana) e `AnaliseEstatisticaService._tratar_outliers_media` (IQR 1.5×, média). A diferença é intencional — parametrizar, não duplicar.

```python
# domain/outliers.py — NOVO
from typing import Literal
import pandas as pd

def tratar_outliers_iqr(
    df: pd.DataFrame,
    coluna: str = "Consumo",
    multiplicador: float = 1.5,
    substituicao: Literal["media", "mediana"] = "mediana",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Detecta outliers pelo método IQR e substitui pelo valor de referência.

    Args:
        multiplicador: 1.5 para classificação (mais sensível), 3.0 para predição (mais conservador)
        substituicao: "media" para classificação estatística, "mediana" para predição
    """
    ...
```

**Arquivos afetados:**
- **[NEW]** `domain/outliers.py`
- **[MODIFY]** `services/analise_estatistica_service.py` — remover `_tratar_outliers_media`, importar `tratar_outliers_iqr(multiplicador=1.5, substituicao="media")`
- **[MODIFY]** `services/predicao_service.py` — remover `_tratar_outliers_mediana`, importar `tratar_outliers_iqr(multiplicador=3.0, substituicao="mediana")`

**Testes impactados:**
- `test_char_tratar_outliers_media_iqr_15` → reaponta para `domain.outliers.tratar_outliers_iqr`
- `test_char_tratar_outliers_mediana_iqr_30` → reaponta para `domain.outliers.tratar_outliers_iqr`
- Os testes de unidade em `PredicaoServiceTests` que chamam `service._tratar_outliers_mediana` diretamente → reaponta para função do domínio

---

#### 1.2 · `domain/bollinger.py` — Protocol de estratégia + implementação padrão

**Problema resolvido:** `_calcular_bandas` é matemática pura vivendo num service. Adicionalmente, o `Protocol EstrategiaBollinger` deixa a porta aberta para trocar o algoritmo (ex: cálculo por dia da semana) sem reescrita do use case.

```python
# domain/bollinger.py — NOVO
from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class EstrategiaBollinger(Protocol):
    """
    Contrato para qualquer algoritmo de cálculo de bandas estatísticas.
    Implemente este protocolo para trocar a estratégia sem modificar o use case.

    Exemplo de alternativas futuras:
    - RollingWindowBollinger (padrão atual — janela de dias consecutivos)
    - DiaSemanasBollinger (média por dia da semana — ex: toda segunda vs toda terça)
    - EWMBollinger (média móvel exponencial — mais peso para dados recentes)
    """
    def calcular(self, df: pd.DataFrame, janela: int) -> pd.DataFrame: ...


class RollingWindowBollinger:
    """
    Estratégia padrão: Bandas de Bollinger com rolling window de dias consecutivos.
    Calcula ±1σ, ±2σ, ±3σ em torno da média móvel.
    """
    def calcular(self, df: pd.DataFrame, janela: int) -> pd.DataFrame:
        df = df.copy()
        df["Média Móvel"] = df["Consumo"].rolling(window=janela, min_periods=1).mean()
        df["Desvio Padrão"] = df["Consumo"].rolling(window=janela, min_periods=1).std()
        df["Banda Inf 3"] = df["Média Móvel"] - 3 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Inf 2"] = df["Média Móvel"] - 2 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Inf 1"] = df["Média Móvel"] - 1 * df["Desvio Padrão"].clip(lower=0)
        df["Banda Sup 1"] = df["Média Móvel"] + 1 * df["Desvio Padrão"]
        df["Banda Sup 2"] = df["Média Móvel"] + 2 * df["Desvio Padrão"]
        df["Banda Sup 3"] = df["Média Móvel"] + 3 * df["Desvio Padrão"]
        return df


# Instância padrão — use cases importam esta, não a classe
estrategia_padrao = RollingWindowBollinger()
```

**Como trocar o algoritmo no futuro sem reescrita:**
```python
# Para cálculo por dia da semana, basta criar a estratégia e injetar:
class DiaSemanasBollinger:
    def calcular(self, df, janela): ...

# No use case:
estatistica_use_case.diario(sensor_id, estrategia=DiaSemanasBollinger())
```

**Arquivos afetados:**
- **[NEW]** `domain/bollinger.py`
- **[MODIFY]** `services/analise_estatistica_service.py` — remover `_calcular_bandas`, importar de `domain.bollinger`

**Testes impactados:**
- `test_char_calcular_bandas_bollinger` → reaponta para `domain.bollinger.RollingWindowBollinger`

---

#### 1.3 · `domain/classificador.py` — regras de classificação por faixa

**Problema resolvido:** `_classificar_consumo_por_faixa` e `_classifica` são regras de negócio nucleares (o que define "normal", "alerta", "crítico") vivendo num service — e sendo acessadas estaticamente por testes. `CLASSIFICATION_METADATA` de `ClassificationRangeService` é a mesma regra numa representação diferente — ambas unificadas aqui.

```python
# domain/classificador.py — NOVO

# Constante de domínio: mapeamento oficial de faixa → semântica de negócio
FAIXAS_METADATA: dict[int, dict] = {
    -2: {"outside_green_range": True,  "severity": "critical", "label": "Consumo Muito Abaixo do Esperado"},
    -1: {"outside_green_range": False, "severity": "green",    "label": "Uso Eficiente"},
     0: {"outside_green_range": False, "severity": "green",    "label": "Consumo Moderado"},
     1: {"outside_green_range": True,  "severity": "warning",  "label": "Uso Elevado"},
     2: {"outside_green_range": True,  "severity": "critical", "label": "Consumo Excessivo"},
}

def classificar_por_faixa(
    consumo: float,
    banda_sup_2: float, banda_sup_1: float,
    banda_inf_1: float, banda_inf_2: float,
) -> int:
    """Regra de negócio pura: mapeia consumo em faixa (-2, -1, 0, 1, 2)."""
    ...

def classificar_serie(df: pd.DataFrame) -> pd.Series:
    """Aplica classificar_por_faixa em cada linha do DataFrame com bandas calculadas."""
    ...
```

**Arquivos afetados:**
- **[NEW]** `domain/classificador.py`
- **[MODIFY]** `services/analise_estatistica_service.py` — remover `_classificar_consumo_por_faixa` e `_classifica`, importar de `domain.classificador`
- **[MODIFY]** `services/classification_range_service.py` — remover `CLASSIFICATION_METADATA`, importar `FAIXAS_METADATA` de `domain.classificador`

**Testes impactados:**
- `test_char_classificar_consumo_por_faixa` → reaponta para `domain.classificador.classificar_por_faixa`

---

#### 1.4 · `domain/ciclo_faturamento.py` — lógica de ciclo mensal

**Problema resolvido:** `_aggregate_monthly` (regra de negócio complexa de faturamento) está na infraestrutura. `_periodos_do_ano` (geração de datas por regras de ciclo) está num service. Ambos são lógica de domínio pura.

**Evidência:** O teste `test_char_aggregate_monthly_ciclo_faturamento` acessa `ExternalDataFetcher._aggregate_monthly` — sinal diagnóstico claro de código de domínio no lugar errado.

```python
# domain/ciclo_faturamento.py — NOVO

def agregar_por_ciclo_mensal(df: pd.DataFrame, dia_inicio_ciclo: int = 1) -> pd.DataFrame:
    """
    Agrega consumo diário em ciclos de faturamento mensais com offset de data.
    Lógica hoje em ExternalDataFetcher._aggregate_monthly.
    """
    ...

def periodos_do_ano(ano: int, dia_fechamento: int) -> list[tuple[date, date]]:
    """
    Gera os 12 períodos de faturamento para o ano dado.
    Lógica hoje em ClassificationHistoryService._periodos_do_ano.
    """
    ...

def safe_date(ano: int, mes: int, dia: int) -> date:
    """Cria date respeitando o último dia do mês."""
    ...
```

**Arquivos afetados:**
- **[NEW]** `domain/ciclo_faturamento.py`
- **[MODIFY]** `infrastructure/db_fetcher.py` — `_aggregate_monthly` passa a delegar para `domain.ciclo_faturamento.agregar_por_ciclo_mensal`
- **[MODIFY]** `services/classification_history_service.py` — `_periodos_do_ano` e `_safe_date` importam de `domain.ciclo_faturamento`

**Testes impactados:**
- `test_char_aggregate_monthly_ciclo_faturamento` → reaponta para `domain.ciclo_faturamento.agregar_por_ciclo_mensal`
- `test_char_periodos_do_ano` → reaponta para `domain.ciclo_faturamento.periodos_do_ano`

---

### FASE 2 — Decompor a infraestrutura

**Objetivo:** `db_fetcher.py` vira três responsabilidades separadas. Exceções de infra ficam isoladas. ModelStore abre caminho para pipelines ML futuros.

---

#### 2.1 · `infrastructure/engine.py` — singleton da engine SQL

```python
# infrastructure/engine.py — NOVO
_engine = None

def get_engine():
    """Module-level Lazy Engine — cria a conexão apenas quando necessário."""
    global _engine
    if _engine is None:
        url = config("EXTERNAL_MYSQL_URL", default=None) or config("EXTERNAL_DB_URL", default=None)
        if not url:
            raise ValueError("EXTERNAL_MYSQL_URL não configurada")
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine
```

**Arquivos afetados:**
- **[NEW]** `infrastructure/engine.py`
- **[MODIFY]** `infrastructure/db_fetcher.py` — remover `_default_engine` e `_get_default_engine()`, importar `get_engine`

---

#### 2.2 · `infrastructure/exceptions.py` + `application/exceptions.py` — hierarquia de exceções por camada

**Problema resolvido:** Views importam exceções de infraestrutura diretamente (`from appSM.infrastructure.db_fetcher import ExternalDataNotFoundError`). A regra é: cada camada tem suas próprias exceções; os use cases fazem a tradução.

```python
# infrastructure/exceptions.py — NOVO
class DataNotFoundError(LookupError):
    """Dado não encontrado no repositório externo. Não vaza para fora da camada de application."""

class DeviceNotFoundError(LookupError):
    """Dispositivo não encontrado. Não vaza para fora da camada de application."""
```

```python
# application/exceptions.py — NOVO
class ConsumoNaoEncontrado(Exception):
    """Não há histórico de consumo para os filtros informados."""
    http_status = 404

class DispositivoNaoEncontrado(Exception):
    """Dispositivo referenciado não existe no sistema."""
    http_status = 404
```

**Fluxo de tradução:**
```
infrastructure lança DataNotFoundError
    → use case captura e relança ConsumoNaoEncontrado
        → view captura ConsumoNaoEncontrado e retorna HTTP 404
```

As views **nunca importam de `infrastructure`**. Os use cases **nunca deixam vazar exceções de infraestrutura**.

**Arquivos afetados:**
- **[NEW]** `infrastructure/exceptions.py`
- **[NEW]** `application/exceptions.py`
- **[MODIFY]** `infrastructure/db_fetcher.py` — `ExternalDataNotFoundError` e `ExternalDeviceNotFoundError` migram para `infrastructure/exceptions.py`; re-exportar no `__init__.py` por compatibilidade temporária
- **[MODIFY]** `api/views.py` — remover imports de `infrastructure`; importar de `application.exceptions`
- **[MODIFY]** Todos os use cases — capturar `infrastructure.exceptions.*`, relançar como `application.exceptions.*`

---

#### 2.3 · `infrastructure/consumo_repository.py` — queries de consumo

Extrai de `db_fetcher.py` as responsabilidades de consulta a `SensorData` e `RelatorioDiarioUnidade`. A classe recebe o nome correto para sua responsabilidade: `ConsumoRepository`.

```python
# infrastructure/consumo_repository.py — NOVO (extraído de db_fetcher.py)
class ConsumoRepository:
    """Repositório read-only de dados de consumo do banco externo."""

    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def buscar_historico_diario(self, sensor_id: str) -> pd.DataFrame:
        """Busca 45 dias de histórico diário para o sensor."""
        ...

    def buscar_historico_mensal_bruto(self, unidade_id: int, inicio, fim) -> pd.DataFrame:
        """Busca dados diários brutos para agregação em ciclos mensais."""
        ...

    def buscar_historico_relatorio_diario(self, unidade_id, data_inicio, data_fim, dias_historico=45) -> pd.DataFrame:
        """Busca histórico diário com contexto para relatórios."""
        ...

    def _carregar_dataframe(self, query, params) -> pd.DataFrame:
        """Executa query, valida e higieniza o resultado."""
        ...
```

**Nota sobre `ExternalDataFetcher`:** O nome `ExternalDataFetcher` é mantido como alias em `infrastructure/__init__.py` durante a fase de migração de testes. Após os testes serem atualizados (Fase 4), o alias é removido.

**Arquivos afetados:**
- **[NEW]** `infrastructure/consumo_repository.py`
- **[MODIFY]** `infrastructure/__init__.py` — exportar `ConsumoRepository`, re-exportar `ExternalDataFetcher` como alias temporário

---

#### 2.4 · `infrastructure/dispositivo_repository.py` — query de dispositivo

Extrai de `db_fetcher.py` as duas queries de `dia_fechamento_fatura` (que hoje estão duplicadas em `_fetch_dia_inicio_ciclo` e `fetch_dispositivo_dia_fechamento`). A deduplicação resolve o item **B-05 do Technical Debt Analysis**.

```python
# infrastructure/dispositivo_repository.py — NOVO (extraído de db_fetcher.py)
class DispositivoRepository:
    """Repositório read-only de metadados de dispositivos."""

    def __init__(self, engine=None):
        self._engine = engine or get_engine()

    def buscar_dia_fechamento(self, dispositivo_id: str) -> int:
        """
        Busca o dia de fechamento da fatura do dispositivo.
        Lança DeviceNotFoundError se o dispositivo não existir.
        Retorna 1 como fallback se dia_fechamento_fatura for NULL.
        """
        ...  # query única, sem duplicação
```

**Arquivos afetados:**
- **[NEW]** `infrastructure/dispositivo_repository.py`
- **[MODIFY]** `infrastructure/db_fetcher.py` — remover `_fetch_dia_inicio_ciclo` e `fetch_dispositivo_dia_fechamento`; usar `DispositivoRepository` internamente durante transição

---

#### 2.5 · `infrastructure/model_store.py` + mover `domain/models/` → `infrastructure/models/`

**Problema resolvido:** Arquivos `.joblib` são artefatos físicos serializados — infraestrutura, não domínio. A classe `LinearRegressionAcumulado` (arquitetura do modelo) permanece em `domain/`, mas os pesos persistidos vão para `infrastructure/`.

```python
# infrastructure/model_store.py — NOVO
from typing import Protocol, Any
from pathlib import Path

class ModelStore(Protocol):
    """
    Contrato para carregamento e salvamento de artefatos de ML.
    Implemente para suportar diferentes backends (disco local, S3, MLflow, etc).

    Uso atual: carregamento de .joblib por client_id (pH classification).
    Uso futuro: qualquer pipeline que precise de modelos persistidos.
    """
    def carregar(self, model_id: str) -> Any: ...
    def salvar(self, model_id: str, modelo: Any) -> None: ...
    def existe(self, model_id: str) -> bool: ...
    def versoes(self, model_id: str) -> list[str]: ...


class LocalJobLibModelStore:
    """
    Implementação padrão: carrega/salva modelos .joblib do disco local.
    Suporta cache em memória com lru_cache para evitar I/O repetido por request.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def carregar(self, model_id: str) -> Any:
        """Carrega modelo com cache em memória (lru_cache por path)."""
        ...

    def salvar(self, model_id: str, modelo: Any) -> None:
        """Persiste modelo em disco. Invalida cache se existir."""
        ...

    def existe(self, model_id: str) -> bool: ...
    def versoes(self, model_id: str) -> list[str]: ...
```

**O `@lru_cache` resolve o C-03 do Technical Debt Analysis** (joblib.load a cada request no pH) de forma generalizada e reutilizável para qualquer modelo futuro.

**Para um novo pipeline de ML**, o padrão fica claro:
```
domain/novo_modelo.py       → arquitetura, treinamento, predição (puro Python)
infrastructure/model_store.py → carregamento/salvamento do artefato
application/novo_use_case.py → orquestra: carrega via store → prediz via domínio
```

**Arquivos afetados:**
- **[NEW]** `infrastructure/model_store.py`
- **[MOVE]** `appSM/domain/models/` → `appSM/infrastructure/models/`
- **[MODIFY]** `projectSM/settings.py` — atualizar `MODELS_DIR` para novo path
- **[MODIFY]** `services/ph_classification_service.py` — usar `LocalJobLibModelStore` (pH fora do escopo, mas a migração de path é necessária)

---

### FASE 3 — Desacoplar as views e dissolver `AnaliseEstatisticaService`

**Objetivo:** Views não conhecem infraestrutura nem algoritmos. `AnaliseEstatisticaService` é dissolvida nos métodos do `EstatisticaUseCase`.

---

#### 3.1 · Criar `application/estatistica_use_case.py` — dissolvendo `AnaliseEstatisticaService`

**Decisão consolidada: dissolução completa.** A classe deixa de existir. Seus algoritmos foram para `domain/` (Fase 1). O que resta são 3 métodos estáticos de orquestração.

```python
# application/estatistica_use_case.py — NOVO (dissolve analise_estatistica_service.py)
from appSM.domain.tratamento import normalizar_historico
from appSM.domain.outliers import tratar_outliers_iqr
from appSM.domain.bollinger import estrategia_padrao, EstrategiaBollinger
from appSM.domain.classificador import classificar_serie, FAIXAS_METADATA
from appSM.infrastructure.consumo_repository import ConsumoRepository
from appSM.infrastructure.exceptions import DataNotFoundError
from appSM.application.exceptions import ConsumoNaoEncontrado


class EstatisticaUseCase:
    """
    Casos de uso de análise estatística de consumo (Bollinger).
    Cada método é um caso de uso independente — a view chama exatamente um.
    """

    @staticmethod
    def diario(sensor_id: str, estrategia: EstrategiaBollinger = None) -> dict:
        """
        Caso de uso: análise estatística diária por sensor.
        Parâmetro estrategia permite injetar algoritmo alternativo de bandas (ex: por dia da semana).
        """
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline(df, janela=30, frequencia="diaria", estrategia=estrategia)

    @staticmethod
    def mensal(unidade_id: int, dispositivo_id: str | None, estrategia: EstrategiaBollinger = None) -> dict:
        """Caso de uso: análise estatística mensal por unidade."""
        try:
            df = ConsumoRepository().buscar_historico_mensal(unidade_id, dispositivo_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline(df, janela=12, frequencia="mensal", estrategia=estrategia)

    @staticmethod
    def dados_completos(sensor_id: str) -> list:
        """Caso de uso: série completa de bandas para visualização gráfica."""
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc

        return _executar_pipeline_completo(df, janela=30, frequencia="diaria")


def _executar_pipeline(df, janela: int, frequencia: str, estrategia=None) -> dict:
    """Pipeline interno de análise estatística — sem estado, sem efeitos colaterais."""
    _estrategia = estrategia or estrategia_padrao

    df = normalizar_historico(df, frequencia=frequencia)
    df_original = df.copy()
    df, mascara_outliers = tratar_outliers_iqr(df, multiplicador=1.5, substituicao="media")
    df = _estrategia.calcular(df, janela=janela)
    df["Classificação"] = classificar_serie(df)

    # Lógica de outlier no último ponto: classificar pelo valor original
    last_idx = df.index[-1]
    if bool(mascara_outliers.loc[last_idx]):
        consumo_original = float(df_original.loc[last_idx, "Consumo"])
        df.loc[last_idx, "Classificação"] = classificar_por_faixa(consumo_original, ...)
        df.loc[last_idx, "Consumo"] = consumo_original

    df = _preencher_nulos(df)
    last_row = df.iloc[-1]
    return {"Data": last_row["Data"], "Consumo": last_row["Consumo"], "Classificação": last_row["Classificação"]}
```

**Arquivos afetados:**
- **[NEW]** `application/estatistica_use_case.py`
- **[DELETE]** `services/analise_estatistica_service.py` — dissolvida

---

#### 3.2 · Criar `application/predicao_use_case.py` — renomear e enxugar `PredicaoService`

```python
# application/predicao_use_case.py — NOVO (renomeia predicao_service.py)
class PredicaoUseCase:

    @staticmethod
    def diario(sensor_id: str) -> float:
        """Caso de uso: predição do próximo consumo diário."""
        try:
            df = ConsumoRepository().buscar_historico_diario(sensor_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc
        return _executar_predicao(df, tipo="diaria", frequencia="diaria")

    @staticmethod
    def mensal(unidade_id: int, dispositivo_id: str | None) -> float:
        """Caso de uso: predição do próximo consumo mensal."""
        try:
            df = ConsumoRepository().buscar_historico_mensal(unidade_id, dispositivo_id)
        except DataNotFoundError as exc:
            raise ConsumoNaoEncontrado(str(exc)) from exc
        return _executar_predicao(df, tipo="mensal", frequencia="mensal")
```

**Arquivos afetados:**
- **[NEW]** `application/predicao_use_case.py`
- **[DELETE]** `services/predicao_service.py`

---

#### 3.3 · Atualizar `application/historico_use_case.py` e `application/range_use_case.py`

- **[RENAME]** `services/classification_history_service.py` → `application/historico_use_case.py`
- **[RENAME]** `services/classification_range_service.py` → `application/range_use_case.py`
- **[MODIFY]** `range_use_case.py` — adicionar injeção de dependência no construtor (Decisão 4 / item M-04): `def __init__(self, historico_use_case=None)`
- **[MODIFY]** Ambos os use cases — capturar `infrastructure.exceptions.*`, relançar `application.exceptions.*`

---

#### 3.4 · Enxugar `api/views.py` — remover toda dependência de infraestrutura

```python
# ANTES — view com 3 camadas de conhecimento
from appSM.infrastructure.db_fetcher import ExternalDataFetcher, ExternalDataNotFoundError, ExternalDeviceNotFoundError
from appSM.services import AnaliseEstatisticaService

class V2AnaliseEstatisticaDiaria(_V2BaseView):
    serializer_class = V2DailySerializer
    is_monthly = False  # ← flag

    def post(self, request):
        validated_data, error = self._validate_payload(request)
        if error: return error
        try:
            historico = self._fetch_history(validated_data)  # ← view chama infra
            resultado = AnaliseEstatisticaService(janela=30).processarDados(historico)  # ← view sabe da janela
            return JsonResponse({"Data": resultado["Data"], ...}, status=200)
        except ExternalDataNotFoundError as exc:  # ← exceção de infra na view
            return JsonResponse({"error": str(exc)}, status=404)

# DEPOIS — view thin com responsabilidade única
from appSM.application.estatistica_use_case import EstatisticaUseCase
from appSM.application.exceptions import ConsumoNaoEncontrado

class V2AnaliseEstatisticaDiaria(_V2BaseView):
    serializer_class = V2DailySerializer
    # sem is_monthly, sem is_anything

    def post(self, request):
        validated_data, error = self._validate_payload(request)
        if error: return error
        try:
            resultado = EstatisticaUseCase.diario(validated_data["sensor_id"])
            return JsonResponse({"Data": resultado["Data"], "Consumo": resultado["Consumo"],
                                 "classificacao": resultado["Classificação"]}, status=200)
        except ConsumoNaoEncontrado as exc:
            return JsonResponse({"error": str(exc)}, status=404)
        except Exception as exc:
            logger.exception("Erro interno: %s", exc)
            return JsonResponse({"error": "Erro interno."}, status=500)
```

**Arquivos afetados:**
- **[MODIFY]** `api/views.py` — remover `_fetch_history`, `is_monthly`, todos imports de `infrastructure`; substituir por chamadas aos use cases; importar apenas de `application.exceptions`
- **[MODIFY]** `application/__init__.py` — exportar os use cases

---

### FASE 4 — Limpeza e reorganização

**Objetivo:** Nomenclatura final, código zumbi removido, testes reorganizados.

---

#### 4.1 · Reorganizar testes em `appSM/tests/`

**Decisão consolidada: pasta própria.**

| Origem | Destino | Conteúdo |
|---|---|---|
| `appSM/tests.py` (classes de API) | `appSM/tests/test_api.py` | `PredictionAndAnalysisAPITests`, `TokenEndpointTests`, `ClassificationRangeAPITests` |
| `appSM/tests.py` (classes de use cases) | `appSM/tests/test_use_cases.py` | `PredicaoServiceTests` → `PredicaoUseCaseTests`, `AnaliseEstatisticaServiceTests` → `EstatisticaUseCaseTests`, `ClassificationHistoryServiceTests` → `HistoricoUseCaseTests` |
| `appSM/test_characterization.py` | `appSM/tests/test_characterization.py` | Mantém conteúdo, atualiza imports |

**Todos os imports atualizados** para refletir as novas localizações de domínio e application.

**Impacto em `test_api.py`:** Os mocks que usam `"appSM.api.views.AnaliseEstatisticaService"` passam para `"appSM.api.views.EstatisticaUseCase"` (o que a view importa agora).

---

#### 4.2 · Remover `MySerializer` e `MySerializerTests`

- Confirmar ausência de chamadores externos ao projeto
- **[MODIFY]** `api/serializers.py` — remover `MySerializer`
- **[MODIFY]** `tests/test_use_cases.py` — remover `MySerializerTests` (ao mover do `tests.py` original, simplesmente não incluir)

---

#### 4.3 · Remover `dataframe_para_historico` e `test_char_dataframe_para_historico_roundtrip`

- Confirmar ausência de chamadores externos
- **[MODIFY]** `infrastructure/consumo_repository.py` — não incluir `dataframe_para_historico` ao extrair de `db_fetcher`
- **[MODIFY]** `infrastructure/__init__.py` — não re-exportar
- **[MODIFY]** `tests/test_characterization.py` — não incluir `test_char_dataframe_para_historico_roundtrip`

---

#### 4.4 · Remover métodos legados de `LinearRegressionAcumulado`

- **[MODIFY]** `domain/regressao_linear.py` — remover `train()` e `prediction()` após confirmar ausência de chamadores

---

#### 4.5 · Remover `JANELA_DIARIA = 7` de `AnaliseEstatisticaService`

- Resolvido automaticamente com a dissolução da classe na Fase 3.1.

---

#### 4.6 · Atualizar `infrastructure/__init__.py` — remover alias legados

Após testes atualizados, remover re-exports de compatibilidade (`ExternalDataFetcher`, `ExternalDataNotFoundError`, `ExternalDeviceNotFoundError`).

---

### FASE 5 — Resiliência

**Objetivo:** Resolver C-02 do Technical Debt Analysis.

> **Esta fase é independente das demais** — pode ser executada antes ou depois das outras fases sem conflito.

#### 5.1 · Cache TTL no `ConsumoRepository`

```python
# infrastructure/consumo_repository.py
from django.core.cache import cache

def buscar_historico_diario(self, sensor_id: str) -> pd.DataFrame:
    cache_key = f"consumo:diario:{sensor_id}:{_local_today():%Y-%m-%d}"
    if (cached := cache.get(cache_key)) is not None:
        return pd.read_json(cached, convert_dates=["Data"])
    df = self._query_diario(sensor_id)
    cache.set(cache_key, df.to_json(), timeout=300)  # 5 min TTL
    return df
```

#### 5.2 · Registrar como ADR

Documentar em `docs/context/ADR/ADR-007-cache-resiliencia.md` a decisão de cache com TTL de 5 minutos como primeira linha de defesa contra intermitência do banco externo — e o gap ainda aberto sobre circuit breaker.

---

## 🔗 Mapa Completo de Testes

| Teste atual | Fase | Ação | Novo destino |
|---|---|---|---|
| `test_char_tratar_outliers_media_iqr_15` | 1.1 | Reaponta | `domain.outliers.tratar_outliers_iqr` |
| `test_char_tratar_outliers_mediana_iqr_30` | 1.1 | Reaponta | `domain.outliers.tratar_outliers_iqr` |
| `test_char_calcular_bandas_bollinger` | 1.2 | Reaponta | `domain.bollinger.RollingWindowBollinger` |
| `test_char_classificar_consumo_por_faixa` | 1.3 | Reaponta | `domain.classificador.classificar_por_faixa` |
| `test_char_aggregate_monthly_ciclo_faturamento` | 1.4 | Reaponta | `domain.ciclo_faturamento.agregar_por_ciclo_mensal` |
| `test_char_periodos_do_ano` | 1.4 | Reaponta | `domain.ciclo_faturamento.periodos_do_ano` |
| `test_char_normalizar_historico_*` | — | Mantém via interface pública | `domain.tratamento.normalizar_historico` |
| `test_char_linear_regression_acumulado_*` | 4.4 | Mantém, remove métodos legados | `domain.regressao_linear.LinearRegressionAcumulado` |
| `test_char_dataframe_para_historico_roundtrip` | 4.3 | **Remover** | — |
| `MySerializerTests` | 4.2 | **Remover** | — |
| `PredicaoServiceTests.*` | 3+4 | Reescrever como `PredicaoUseCaseTests` | `tests/test_use_cases.py` |
| `AnaliseEstatisticaServiceTests.*` | 3+4 | Reescrever como `EstatisticaUseCaseTests` | `tests/test_use_cases.py` |
| `ClassificationHistoryServiceTests.*` | 3+4 | Renomear como `HistoricoUseCaseTests` | `tests/test_use_cases.py` |
| `PredictionAndAnalysisAPITests.*` (mocks) | 3+4 | Atualizar paths de mock | `tests/test_api.py` |
| `TokenEndpointTests` | 4 | Mover sem alteração | `tests/test_api.py` |
| `ClassificationRangeAPITests` | 3+4 | Atualizar mocks | `tests/test_api.py` |

---

## 📊 Resumo de Impacto por Fase

| Fase | Novos arquivos | Modificados | Removidos | Testes impactados |
|---|---|---|---|---|
| 1 — Domínio | 4 (`outliers`, `bollinger`, `classificador`, `ciclo_faturamento`) | 3 | 0 | 6 de char |
| 2 — Infra | 5 (`engine`, `exceptions×2`, `consumo_repo`, `dispositivo_repo`, `model_store`) | 1 (`db_fetcher`) | 0 | 1 de char |
| 3 — Desacoplamento | 3 (`estatistica_uc`, `predicao_uc`, `app_exceptions`) | `views`, 2 use cases | 2 (`analise_*`, `predicao_*`) | Todos API tests |
| 4 — Limpeza | 1 (`tests/__init__`) | Todos imports | 3 (legados + `db_fetcher`) | ~12 testes (reorganização) |
| 5 — Resiliência | 1 (ADR-007) | `consumo_repository` | 0 | 0 |

---

## 🧭 Ordem de Execução Recomendada

A ordem das sub-tarefas dentro de cada fase pode variar, mas **as fases devem ser executadas em sequência** — cada uma deixa o sistema em estado funcionalmente correto (testes passando) antes da próxima.

```
Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5 (qualquer momento)
```

**Checkpoint após cada fase:** executar `python manage.py test` e confirmar que todos os testes passam antes de avançar.

---

*Versão final — 2026-08-05. Decisões consolidadas. Pronto para execução por agente implementador.*
