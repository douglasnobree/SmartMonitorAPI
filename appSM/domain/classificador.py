import pandas as pd

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
    if consumo >= banda_sup_2:
        return 2
    if banda_sup_2 > consumo >= banda_sup_1:
        return 1
    if banda_sup_1 > consumo >= banda_inf_1:
        return 0
    if banda_inf_1 > consumo >= banda_inf_2:
        return -1
    return -2

def classificar_serie(df: pd.DataFrame) -> pd.Series:
    """Aplica classificar_por_faixa em cada linha do DataFrame com bandas calculadas."""
    def _classifica(row):
        c = row["Consumo"]
        bs2 = row.get("Banda Sup 2")
        bs1 = row.get("Banda Sup 1")
        bi1 = row.get("Banda Inf 1")
        bi2 = row.get("Banda Inf 2")
        sd = row.get("Desvio Padrão")
        
        if pd.isna(sd):
            return None
            
        return classificar_por_faixa(c, bs2, bs1, bi1, bi2)
        
    return df.apply(_classifica, axis=1)
