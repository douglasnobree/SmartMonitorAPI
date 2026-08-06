import calendar
import pandas as pd
from datetime import date, timedelta

def agregar_por_ciclo_mensal(df: pd.DataFrame, dia_inicio_ciclo: int = 1) -> pd.DataFrame:
    """
    Agrega consumo diário em ciclos de faturamento mensais com offset de data.
    """
    working = df.copy()
    offset_days = dia_inicio_ciclo - 1
    if offset_days > 0:
        working.index = working.index - pd.Timedelta(days=offset_days)
        
    mensal = working.sort_index()[["Consumo"]].resample("MS").sum()
    
    if mensal.empty:
        raise ValueError("Nenhum registro de consumo consolidado.")
        
    if offset_days > 0:
        mensal.index = mensal.index + pd.Timedelta(days=offset_days)

    return mensal

def safe_date(ano: int, mes: int, dia: int) -> date:
    """Cria date respeitando o último dia do mês."""
    ultimo_dia = calendar.monthrange(ano, mes)[1]
    return date(ano, mes, min(dia, ultimo_dia))

def periodos_do_ano(ano: int, dia_fechamento: int) -> list[tuple[date, date]]:
    """
    Gera os 12 períodos de faturamento para o ano dado.
    """
    periodos = []
    for mes in range(1, 13):
        if dia_fechamento <= 1:
            inicio = safe_date(ano, mes, 1)
            proximo_mes = inicio + pd.DateOffset(months=1)
            fim = proximo_mes.date() - timedelta(days=1)
        else:
            fim_base = safe_date(ano, mes, dia_fechamento)
            inicio = (pd.Timestamp(fim_base) - pd.DateOffset(months=1)).date()
            fim = fim_base - timedelta(days=1)
        periodos.append((inicio, fim))
    return periodos
