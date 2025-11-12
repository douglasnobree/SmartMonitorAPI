"""
Serviço unificado de análise estatística usando Bandas de Bollinger.

Este módulo implementa a classificação de consumo (diário ou mensal) baseada em
análise estatística utilizando o método de Bandas de Bollinger.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

from ml_pipeline.Tratamento import Tratamento

# Configure logger
logger = logging.getLogger(__name__)


class AnaliseEstatisticaService(Tratamento):
    """
    Serviço unificado para classificação de consumo usando Bandas de Bollinger.
    
    Analisa os dados de consumo e classifica o último registro em uma de 5 categorias
    baseadas nas bandas de Bollinger calculadas com janela móvel configurável.
    
    Classificações:
        -2: Faixa inferior 2 (muito abaixo do normal)
        -1: Faixa inferior 1 (abaixo do normal)
         0: Faixa ideal (normal)
         1: Faixa superior 1 (acima do normal)
         2: Faixa superior 2 (muito acima do normal)
    
    Uso:
        # Análise diária (janela de 30 dias)
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)
        
        # Análise mensal (janela de 12 meses)
        service = AnaliseEstatisticaService(janela=12)
        resultado = service.processarDados(dados)
    """
    
    # Constantes para janelas padrão
    JANELA_DIARIA = 30
    JANELA_MENSAL = 12
    
    def __init__(self, janela: Optional[int] = None):
        """
        Inicializa o serviço de análise estatística.
        
        Args:
            janela (int, optional): Tamanho da janela para média móvel e desvio padrão.
                                   Se None, usa JANELA_DIARIA (30) como padrão.
        """
        self.janela = janela if janela is not None else self.JANELA_DIARIA
        logger.info(f"AnaliseEstatisticaService inicializado com janela={self.janela}")
    
    def processarDados(self, dados_request: Dict[str, float]) -> Dict[str, Any]:
        """
        Processa dados de consumo e retorna a classificação do último registro.
        
        Args:
            dados_request (dict): Dicionário com datas e consumos.
                                 Formato: {'DD/MM/YYYY': valor_float}
        
        Returns:
            dict: Dicionário contendo:
                - Data (str): Data do último registro
                - Consumo (float): Valor do consumo
                - Classificação (int ou str): Classificação do consumo
        
        Raises:
            ValueError: Se dados_request estiver vazio ou inválido
            Exception: Se houver erro no processamento dos dados
        """
        try:
            if not dados_request:
                raise ValueError("dados_request não pode estar vazio")
            
            # Criação do DataFrame
            df = pd.DataFrame({
                'Data': list(dados_request.keys()), 
                'Consumo': list(dados_request.values())
            })
            
            logger.debug(f"DataFrame criado com {len(df)} registros")
            
            # Cálculo da média móvel
            df["Média Móvel"] = (
                df["Consumo"]
                .rolling(window=self.janela, min_periods=1)
                .mean()
            )
            
            # Cálculo do desvio padrão
            df["Desvio Padrão"] = (
                df["Consumo"]
                .rolling(window=self.janela, min_periods=1)
                .std()
            )
            
            # Cálculo das bandas de Bollinger
            df["Banda Inf 3"] = df["Média Móvel"] - 3 * df["Desvio Padrão"]
            df["Banda Inf 2"] = df["Média Móvel"] - 2 * df["Desvio Padrão"]
            df["Banda Inf 1"] = df["Média Móvel"] - 1 * df["Desvio Padrão"]
            df["Banda Sup 1"] = df["Média Móvel"] + 1 * df["Desvio Padrão"]
            df["Banda Sup 2"] = df["Média Móvel"] + 2 * df["Desvio Padrão"]
            df["Banda Sup 3"] = df["Média Móvel"] + 3 * df["Desvio Padrão"]
            
            # Aplicar classificação
            df["Classificação"] = df.apply(self._classifica, axis=1)
            
            # Preencher valores nulos
            for col in df.columns:
                if col != 'Data' and col != 'Classificação':
                    df[col] = df[col].fillna(0)
                elif col == 'Classificação':
                    df[col] = df[col].fillna("Sem classificação")
            
            # Obter último registro
            last_row = df.iloc[-1]
            classification = {
                "Data": last_row['Data'],
                "Consumo": last_row['Consumo'],
                "Classificação": last_row['Classificação']
            }
            
            logger.debug(
                f"Classificação calculada para {last_row['Data']}: "
                f"{classification['Classificação']}"
            )
            
            return classification
            
        except ValueError as e:
            logger.error(f"Erro de validação: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Erro em AnaliseEstatisticaService: {str(e)}")
            raise Exception(str(e))
    
    @staticmethod
    def _classifica(row: pd.Series) -> Optional[int]:
        """
        Classifica o consumo baseado nas bandas de Bollinger.
        
        Args:
            row (pd.Series): Linha do DataFrame com valores calculados
        
        Returns:
            int ou None: Classificação numérica ou None se sem dados suficientes
        """
        c = row["Consumo"]
        bs2, bs1, sd, bi1, bi2 = (
            row["Banda Sup 2"],
            row["Banda Sup 1"],
            row["Desvio Padrão"],
            row["Banda Inf 1"],
            row["Banda Inf 2"]
        )
        
        # Sem dados suficientes para classificar
        if pd.isna(sd):
            return None
        
        # Classificação baseada nas bandas
        if c >= bs2:
            return 2  # Muito acima do normal
        elif bs2 > c >= bs1:
            return 1  # Acima do normal
        elif bs1 > c >= bi1:
            return 0  # Normal (ideal)
        elif bi1 > c >= bi2:
            return -1  # Abaixo do normal
        else:
            return -2  # Muito abaixo do normal


# Aliases para manter compatibilidade com código existente
class analiseEstatisticaDiaria_service(AnaliseEstatisticaService):
    """Alias para análise diária - mantém compatibilidade."""
    
    def __init__(self):
        super().__init__(janela=self.JANELA_DIARIA)


class analiseEstatisticaMensal_service(AnaliseEstatisticaService):
    """Alias para análise mensal - mantém compatibilidade."""
    
    def __init__(self):
        super().__init__(janela=self.JANELA_MENSAL)
