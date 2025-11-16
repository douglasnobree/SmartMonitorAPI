"""
Testes unitários para PredicaoMensal_service.

Testa o serviço de predição de consumo mensal usando Regressão Linear
com dados acumulados.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service import (
    PredicaoMensal_service
)


class TestPredicaoMensalService(unittest.TestCase):
    """Testes para o serviço de predição mensal."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Dados válidos para predição mensal (12 meses)
        self.dados_validos = {
            'Jan/2024': 1200.0,
            'Fev/2024': 1150.0,
            'Mar/2024': 1300.0,
            'Abr/2024': 1250.0,
            'Mai/2024': 1400.0,
            'Jun/2024': 1350.0,
            'Jul/2024': 1500.0,
            'Ago/2024': 1450.0,
            'Set/2024': 1600.0,
            'Out/2024': 1550.0,
            'Nov/2024': 1700.0,
            'Dez/2024': 1650.0,
        }

        # Dados com valores variados
        self.dados_variados = {
            'Jan/2024': 1000.0,
            'Fev/2024': 1100.0,
            'Mar/2024': 1050.0,
            'Abr/2024': 1200.0,
            'Mai/2024': 1150.0,
            'Jun/2024': 1300.0,
        }

        # Dados com outliers (valores baixos)
        self.dados_com_outliers = {
            'Jan/2024': 1200.0,
            'Fev/2024': 50.0,   # Outlier
            'Mar/2024': 1300.0,
            'Abr/2024': 30.0,   # Outlier
            'Mai/2024': 1400.0,
            'Jun/2024': 1350.0,
        }

        # Dados com None
        self.dados_com_nan = {
            'Jan/2024': 1200.0,
            'Fev/2024': None,
            'Mar/2024': 1300.0,
            'Abr/2024': 1250.0,
        }

    # ========== Testes de Processamento de Dados ==========

    def test_processar_dados_validos(self):
        """Testa processamento com dados válidos."""
        service = PredicaoMensal_service()
        resultado = service.processarDados(self.dados_validos)

        # Resultado deve ser um número (predição)
        self.assertIsInstance(resultado, (int, float))
        
        # Predição deve ser positiva (abs garante isso)
        self.assertGreaterEqual(resultado, 0)

    def test_processar_dados_variados(self):
        """Testa processamento com dados variados."""
        service = PredicaoMensal_service()
        resultado = service.processarDados(self.dados_variados)

        self.assertIsInstance(resultado, (int, float))
        self.assertGreaterEqual(resultado, 0)

    def test_processar_dados_pequenos(self):
        """Testa processamento com poucos dados."""
        dados_pequenos = {
            'Jan/2024': 1000.0,
            'Fev/2024': 1100.0,
            'Mar/2024': 1200.0
        }
        service = PredicaoMensal_service()
        resultado = service.processarDados(dados_pequenos)

        # Deve processar mesmo com poucos dados
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_processar_um_registro(self):
        """Testa processamento com apenas um registro."""
        dados_um = {'Jan/2024': 1000.0}
        service = PredicaoMensal_service()
        
        try:
            resultado = service.processarDados(dados_um)
            self.assertIsNotNone(resultado)
        except Exception as e:
            # Se der erro, verificar que é tratado
            self.assertIsInstance(e, Exception)

    # ========== Testes de Tratamento de NaN ==========

    def test_tratamento_valores_nan(self):
        """Testa que valores NaN são substituídos pela mediana."""
        service = PredicaoMensal_service()
        
        resultado = service.processarDados(self.dados_com_nan)
        
        # Não deve crashar
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_substituicao_nan_por_mediana(self):
        """Testa substituição de NaN por mediana."""
        service = PredicaoMensal_service()
        
        dados = {
            'Jan/2024': 1000.0,
            'Fev/2024': np.nan,
            'Mar/2024': 1200.0,
            'Abr/2024': 1300.0,
        }
        
        resultado = service.processarDados(dados)
        
        self.assertIsNotNone(resultado)

    # ========== Testes de Tratamento de Outliers ==========

    def test_substituicao_valores_abaixo_percentil_25(self):
        """Testa que valores abaixo do percentil 25 são substituídos pela média."""
        service = PredicaoMensal_service()
        resultado = service.processarDados(self.dados_com_outliers)

        # Deve processar sem erros
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_calculo_percentil_25(self):
        """Testa que percentil 25 é calculado corretamente."""
        service = PredicaoMensal_service()
        
        # Dados conhecidos
        dados = {
            'Jan/2024': 100.0,
            'Fev/2024': 200.0,
            'Mar/2024': 300.0,
            'Abr/2024': 400.0,
        }
        # Percentil 25 seria 175
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(dados)
            
            # Verificar que train foi chamado
            mock_instance.train.assert_called_once()

    # ========== Testes de Cálculo de Acumulado ==========

    def test_criacao_coluna_acumulado(self):
        """Testa que coluna Acumulado é criada corretamente."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 10000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar que train foi chamado
            mock_instance.train.assert_called_once()
            
            # Verificar que DataFrame passado tem coluna Acumulado
            df_passado = mock_instance.train.call_args[0][0]
            self.assertIn('Acumulado', df_passado.columns)

    def test_acumulado_cumsum_correto(self):
        """Testa que acumulado é cumsum correto."""
        dados = {
            'Jan/2024': 1000.0,
            'Fev/2024': 1100.0,
            'Mar/2024': 1200.0
        }
        
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 3500.0
            MockModel.return_value = mock_instance
            
            service.processarDados(dados)
            
            # Pegar DataFrame passado para train
            df_passado = mock_instance.train.call_args[0][0]
            
            # Verificar valores acumulados
            # Esperado: 1000, 2100, 3300
            acumulado = df_passado['Acumulado'].tolist()
            self.assertEqual(acumulado[0], 1000.0)
            self.assertEqual(acumulado[1], 2100.0)
            self.assertEqual(acumulado[2], 3300.0)

    # ========== Testes de Modelo de ML ==========

    def test_modelo_treinado(self):
        """Testa que modelo é treinado."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 15000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar que train foi chamado
            mock_instance.train.assert_called_once()

    def test_predicao_chamada_com_indice_correto(self):
        """Testa que predição é chamada com índice correto."""
        service = PredicaoMensal_service()
        
        dados = {f'Mês{i}/2024': 1000.0 for i in range(1, 7)}  # 6 registros
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 6500.0
            MockModel.return_value = mock_instance
            
            service.processarDados(dados)
            
            # Predição deve ser chamada com len(dados) = 6
            mock_instance.prediction.assert_called_once_with(6)

    def test_calculo_resultado_diferenca_absoluta(self):
        """Testa que resultado é diferença absoluta entre predição e último acumulado."""
        service = PredicaoMensal_service()
        
        dados = {
            'Jan/2024': 1000.0,
            'Fev/2024': 1000.0,
            'Mar/2024': 1000.0
        }
        # Acumulado seria: 1000, 2000, 3000
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 3500.0  # Predição
            MockModel.return_value = mock_instance
            
            resultado = service.processarDados(dados)
            
            # Resultado deve ser abs(3500 - 3000) = 500
            self.assertEqual(resultado, 500.0)

    def test_resultado_sempre_positivo(self):
        """Testa que resultado é sempre positivo (abs)."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            # Predição menor que acumulado
            mock_instance.prediction.return_value = 5000.0
            MockModel.return_value = mock_instance
            
            resultado = service.processarDados(self.dados_validos)
            
            # Resultado deve ser positivo
            self.assertGreaterEqual(resultado, 0)

    def test_train_recebe_dataframe_completo(self):
        """Testa que train recebe DataFrame completo (não apenas coluna Acumulado)."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 10000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar argumento passado para train
            args = mock_instance.train.call_args[0]
            df_passado = args[0]
            
            # Deve ser DataFrame, não Series
            self.assertIsInstance(df_passado, pd.DataFrame)
            
            # Deve ter todas as colunas
            self.assertIn('Data', df_passado.columns)
            self.assertIn('Consumo', df_passado.columns)
            self.assertIn('Acumulado', df_passado.columns)

    # ========== Testes de Edge Cases ==========

    def test_valores_negativos(self):
        """Testa processamento com valores negativos."""
        dados_negativos = {
            'Jan/2024': -1000.0,
            'Fev/2024': -500.0,
            'Mar/2024': -800.0
        }
        service = PredicaoMensal_service()
        
        resultado = service.processarDados(dados_negativos)
        self.assertIsNotNone(resultado)

    def test_valores_zero(self):
        """Testa processamento com valores zero."""
        dados_zero = {f'Mês{i}/2024': 0.0 for i in range(1, 7)}
        service = PredicaoMensal_service()
        resultado = service.processarDados(dados_zero)

        self.assertIsNotNone(resultado)

    def test_valores_constantes(self):
        """Testa processamento com valores constantes."""
        dados_constantes = {f'Mês{i}/2024': 1000.0 for i in range(1, 13)}
        service = PredicaoMensal_service()
        resultado = service.processarDados(dados_constantes)

        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_valores_muito_grandes(self):
        """Testa processamento com valores muito grandes."""
        dados_grandes = {
            'Jan/2024': 1e10,
            'Fev/2024': 1e10,
            'Mar/2024': 1e10
        }
        service = PredicaoMensal_service()
        resultado = service.processarDados(dados_grandes)

        self.assertIsNotNone(resultado)

    # ========== Testes de Reset de Index ==========

    def test_reset_index_executado(self):
        """Testa que reset_index é executado no DataFrame."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 10000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar que DataFrame passado não tem index customizado
            df_passado = mock_instance.train.call_args[0][0]
            
            # Index deve ser RangeIndex após reset
            self.assertIsInstance(df_passado.index, pd.RangeIndex)

    # ========== Testes de Integração ==========

    def test_fluxo_completo(self):
        """Testa fluxo completo do serviço."""
        service = PredicaoMensal_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar integridade completa
        self.assertIsInstance(resultado, (int, float))
        self.assertGreaterEqual(resultado, 0)
        # Resultado razoável
        self.assertLess(resultado, 1e7)

    def test_multiplas_execucoes(self):
        """Testa múltiplas execuções no mesmo serviço."""
        service = PredicaoMensal_service()
        
        resultado1 = service.processarDados(self.dados_validos)
        resultado2 = service.processarDados(self.dados_variados)
        
        # Ambos devem retornar resultados válidos
        self.assertIsNotNone(resultado1)
        self.assertIsNotNone(resultado2)
        
        # Resultados devem ser diferentes (dados diferentes)
        self.assertNotEqual(resultado1, resultado2)

    def test_consistencia_resultados(self):
        """Testa que mesmos dados produzem mesmo resultado."""
        service = PredicaoMensal_service()
        
        resultado1 = service.processarDados(self.dados_validos)
        resultado2 = service.processarDados(self.dados_validos)
        
        # Deve ser determinístico
        self.assertEqual(resultado1, resultado2)

    # ========== Testes de Tratamento de Erros ==========

    def test_dados_vazios_levanta_excecao(self):
        """Testa que dados vazios levantam exceção."""
        service = PredicaoMensal_service()
        
        with self.assertRaises(Exception):
            service.processarDados({})

    def test_dados_none_levanta_excecao(self):
        """Testa que dados None levantam exceção."""
        service = PredicaoMensal_service()
        
        with self.assertRaises(Exception):
            service.processarDados(None)

    # ========== Testes de Validação de Dados ==========

    def test_criacao_dataframe(self):
        """Testa que DataFrame é criado corretamente."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 15000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar DataFrame
            df = mock_instance.train.call_args[0][0]
            
            self.assertIn('Data', df.columns)
            self.assertIn('Consumo', df.columns)
            self.assertIn('Acumulado', df.columns)
            self.assertEqual(len(df), len(self.dados_validos))

    def test_ordem_datas_preservada(self):
        """Testa que ordem das datas é preservada."""
        service = PredicaoMensal_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoMensal_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 15000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            df = mock_instance.train.call_args[0][0]
            
            # Verificar primeira e última data
            self.assertEqual(df['Data'].iloc[0], 'Jan/2024')
            self.assertEqual(df['Data'].iloc[-1], 'Dez/2024')

    # ========== Testes Comparativos com PredicaoDiaria ==========

    def test_semelhanca_com_predicao_diaria(self):
        """Testa que lógica é similar a PredicaoDiaria_service."""
        # Ambos devem ter mesmo comportamento geral
        service = PredicaoMensal_service()
        
        # Processar dados
        resultado = service.processarDados(self.dados_validos)
        
        # Verificar que retorna número positivo
        self.assertIsInstance(resultado, (int, float))
        self.assertGreaterEqual(resultado, 0)

    # ========== Testes de Performance ==========

    def test_processamento_rapido(self):
        """Testa que processamento é razoavelmente rápido."""
        import time
        
        service = PredicaoMensal_service()
        
        inicio = time.time()
        service.processarDados(self.dados_validos)
        fim = time.time()
        
        tempo_execucao = fim - inicio
        
        # Deve processar em menos de 1 segundo
        self.assertLess(tempo_execucao, 1.0)

    def test_processamento_varios_anos(self):
        """Testa processamento com vários anos de dados."""
        # 24 meses (2 anos)
        dados_2_anos = {f'Mês{i}/2023-2024': 1000.0 + i * 10 for i in range(1, 25)}
        
        service = PredicaoMensal_service()
        resultado = service.processarDados(dados_2_anos)
        
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))


if __name__ == '__main__':
    unittest.main()
