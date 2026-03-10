"""
Modelo de Regressão Linear com valores acumulados.

Implementa predição de consumo usando regressão linear sobre dados acumulados.
Encapsula o pré-processamento necessário para este modelo específico.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ml_pipeline.modelos.base_modelo import ModeloPredicao


class LinearRegression_Acumulado(ModeloPredicao):
    """
    Modelo de Regressão Linear que utiliza valores acumulados.
    
    Este modelo:
    1. Preenche valores nulos com a mediana
    2. Cria coluna 'Acumulado' com cumsum do consumo
    3. Treina regressão linear sobre os valores acumulados
    4. Prevê consumo incremental com ajustes baseados no tipo de predição
    
    Args:
        tipo_predicao: Tipo de predição ('diaria', 'mensal', ou None para simples)
    """
    
    def __init__(self, tipo_predicao: str = None):
        self.model = None
        self.df_processado = None
        self.tipo_predicao = tipo_predicao

    def treinar(self, dados_brutos: pd.DataFrame) -> None:
        if 'Consumo' not in dados_brutos.columns:
            raise ValueError("DataFrame deve conter coluna 'Consumo'")
        
        # Fazer cópia para não modificar dados originais
        df = dados_brutos.copy()
        
        # Pré-processamento específico deste modelo
        df['Consumo'].fillna(df['Consumo'].median(), inplace=True)
        df['Acumulado'] = df['Consumo'].cumsum()
        df.reset_index(inplace=True, drop=True)
        
        # Treinar modelo de regressão linear
        x = np.arange(len(df)).reshape(-1, 1)
        y = np.array(df['Acumulado'])
        
        self.model = LinearRegression()
        self.model.fit(x, y)
        self.df_processado = df

    def prever(self, n_passos: int) -> float:
        """
        Prevê consumo incremental para n_passos à frente.
        
        Aplica automaticamente ajustes baseados no tipo_predicao configurado
        no construtor do modelo.
        
        Args:
            n_passos: Índice para predição (número de registros)
        
        Returns:
            Consumo incremental previsto (não-acumulado) com ajuste aplicado
        
        Raises:
            ValueError: Se modelo não foi treinado
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado.")
        
        # Predição base: valor acumulado
        proximo_acumulado = self.model.predict([[n_passos]])[0]
        acumulado_atual = self.df_processado['Acumulado'].iloc[-1]
        
        # Se não há tipo configurado, retorna predição simples (incremental)
        if self.tipo_predicao is None:
            return proximo_acumulado - acumulado_atual
        
        # Aplicar ajustes específicos por tipo de predição
        match self.tipo_predicao:
            case 'diaria':
                # Ajuste para predição diária
                if proximo_acumulado < acumulado_atual:
                    # Se predição acumulada é menor que atual, aplicar correção
                    previsao_anterior = self.model.predict([[n_passos - 1]])[0]
                    media_acumulada = (acumulado_atual + previsao_anterior) / 2
                    resultado = proximo_acumulado - media_acumulada
                else:
                    resultado = proximo_acumulado - acumulado_atual
                
                # Garantir que não seja negativo
                return max(resultado, 0)
            
            case 'mensal':
                # Ajuste estatístico para predição mensal
                # Calcular resíduos do modelo no histórico
                indices_historico = list(range(len(self.df_processado)))
                pred_historico = self.model.predict(np.array(indices_historico).reshape(-1, 1))
                residuos = pred_historico - self.df_processado['Acumulado'].values
                
                # Estatísticas dos resíduos
                sigma_erro = np.std(residuos)
                media_erro = np.mean(residuos)
                
                # Aplicar ajuste: remover viés e adicionar 1 sigma
                previsao_ajustada = proximo_acumulado - media_erro + 1 * sigma_erro
                
                # Usar abs porque previsão pode ser menor que acumulado
                return abs(previsao_ajustada - acumulado_atual)
            
            case _:
                # Caso padrão: qualquer outro tipo retorna predição simples
                return proximo_acumulado - acumulado_atual
    
    # Métodos legados para compatibilidade (deprecated)
    def train(self, data: pd.DataFrame) -> None:
        """Método legado. Use treinar() ao invés."""
        self.treinar(data)
    
    def prediction(self, indices: int | list[int]) -> int | np.ndarray:
        """
        Método legado para predição de valores acumulados.
        Use prever() para nova interface.
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado.")

        pred = self.model.predict(np.array(indices).reshape(-1, 1))

        if isinstance(indices, int):
            return pred[0]
        
        return pred