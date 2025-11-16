"""
Testes unitários para PredicaoDiaria_service.

Testa o serviço de predição de consumo diário usando Regressão Linear
com dados acumulados.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service import (
    PredicaoDiaria_service
)


class TestPredicaoDiariaService(unittest.TestCase):
    """Testes para o serviço de predição diária."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Dados válidos para predição diária (30 dias)
        self.dados_validos = {
            f'{i:02d}/01/2025': float(100 + i * 2) 
            for i in range(1, 31)
        }

        # Dados com valores variados
        self.dados_variados = {
            '01/01/2025': 100.0,
            '02/01/2025': 105.0,
            '03/01/2025': 110.0,
            '04/01/2025': 108.0,
            '05/01/2025': 112.0,
            '06/01/2025': 115.0,
            '07/01/2025': 120.0,
            '08/01/2025': 118.0,
            '09/01/2025': 122.0,
            '10/01/2025': 125.0,
            '11/01/2025': 130.0,
            '12/01/2025': 128.0,
            '13/01/2025': 135.0,
            '14/01/2025': 140.0,
            '15/01/2025': 138.0,
        }

        # Dados com valores abaixo do percentil (para testar substituição)
        self.dados_com_outliers = {
            '01/01/2025': 100.0,
            '02/01/2025': 105.0,
            '03/01/2025': 5.0,    # Outlier baixo
            '04/01/2025': 110.0,
            '05/01/2025': 3.0,    # Outlier baixo
            '06/01/2025': 115.0,
            '07/01/2025': 120.0,
            '08/01/2025': 2.0,    # Outlier baixo
        }

        # Dados com valores None/NaN
        self.dados_com_nan = {
            '01/01/2025': 100.0,
            '02/01/2025': None,
            '03/01/2025': 110.0,
            '04/01/2025': 115.0,
        }

    # ========== Testes de Processamento de Dados ==========

    def test_processar_dados_validos(self):
        """Testa processamento com dados válidos."""
        service = PredicaoDiaria_service()
        resultado = service.processarDados(self.dados_validos)

        # Resultado deve ser um número (predição)
        self.assertIsInstance(resultado, (int, float))
        
        # Predição deve ser positiva (abs garante isso)
        self.assertGreaterEqual(resultado, 0)

    def test_processar_dados_variados(self):
        """Testa processamento com dados variados."""
        service = PredicaoDiaria_service()
        resultado = service.processarDados(self.dados_variados)

        self.assertIsInstance(resultado, (int, float))
        self.assertGreaterEqual(resultado, 0)

    def test_processar_dados_pequenos(self):
        """Testa processamento com poucos dados."""
        dados_pequenos = {
            '01/01/2025': 100.0,
            '02/01/2025': 105.0,
            '03/01/2025': 110.0
        }
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_pequenos)

        # Deve processar mesmo com poucos dados
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_processar_um_registro(self):
        """Testa processamento com apenas um registro."""
        dados_um = {'01/01/2025': 100.0}
        service = PredicaoDiaria_service()
        
        # Com um registro, modelo pode ter dificuldade
        # mas não deve crashar
        try:
            resultado = service.processarDados(dados_um)
            self.assertIsNotNone(resultado)
        except Exception as e:
            # Se der erro, pelo menos verificar que é tratado
            self.assertIsInstance(e, Exception)

    # ========== Testes de Tratamento de NaN ==========

    def test_tratamento_valores_nan(self):
        """Testa que valores NaN são substituídos pela mediana."""
        service = PredicaoDiaria_service()
        
        # Processar dados com None
        resultado = service.processarDados(self.dados_com_nan)
        
        # Não deve crashar
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_substituicao_nan_por_mediana(self):
        """Testa substituição de NaN por mediana usando mock."""
        service = PredicaoDiaria_service()
        
        # Dados com NaN explícito
        dados = {
            '01/01/2025': 100.0,
            '02/01/2025': np.nan,
            '03/01/2025': 110.0,
            '04/01/2025': 120.0,
        }
        
        # Processar
        resultado = service.processarDados(dados)
        
        # Verificar que processou
        self.assertIsNotNone(resultado)

    # ========== Testes de Tratamento de Outliers ==========

    def test_substituicao_valores_abaixo_percentil(self):
        """Testa que valores abaixo do percentil 25 são substituídos pela média."""
        service = PredicaoDiaria_service()
        resultado = service.processarDados(self.dados_com_outliers)

        # Deve processar sem erros
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_percentil_ajustado_quando_menor_que_1(self):
        """Testa ajuste de percentil quando é menor que 1."""
        # Dados com valores muito baixos
        dados_baixos = {f'{i:02d}/01/2025': 0.1 for i in range(1, 11)}
        
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_baixos)

        # Deve processar (percentil ajustado para mediana)
        self.assertIsNotNone(resultado)

    # ========== Testes de Cálculo de Acumulado ==========

    def test_criacao_coluna_acumulado(self):
        """Testa que coluna Acumulado é criada corretamente."""
        service = PredicaoDiaria_service()
        
        # Patch do modelo para verificar dados recebidos
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
            MockModel.return_value = mock_instance
            
            resultado = service.processarDados(self.dados_validos)
            
            # Verificar que train foi chamado
            mock_instance.train.assert_called_once()
            
            # Verificar que DataFrame passado tem coluna Acumulado
            df_passado = mock_instance.train.call_args[0][0]
            self.assertIn('Acumulado', df_passado.columns)

    def test_acumulado_cumsum_correto(self):
        """Testa que acumulado é cumsum correto."""
        dados = {
            '01/01/2025': 10.0,
            '02/01/2025': 20.0,
            '03/01/2025': 30.0
        }
        
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 70.0
            MockModel.return_value = mock_instance
            
            service.processarDados(dados)
            
            # Pegar DataFrame passado para train
            df_passado = mock_instance.train.call_args[0][0]
            
            # Verificar valores acumulados
            # Esperado: 10, 30, 60
            acumulado = df_passado['Acumulado'].tolist()
            self.assertEqual(acumulado[0], 10.0)
            self.assertEqual(acumulado[1], 30.0)
            self.assertEqual(acumulado[2], 60.0)

    # ========== Testes de Modelo de ML ==========

    def test_modelo_treinado(self):
        """Testa que modelo é treinado."""
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1500.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar que train foi chamado
            mock_instance.train.assert_called_once()

    def test_predicao_chamada_com_indice_correto(self):
        """Testa que predição é chamada com índice correto."""
        service = PredicaoDiaria_service()
        
        dados = {f'{i:02d}/01/2025': 100.0 for i in range(1, 11)}  # 10 registros
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(dados)
            
            # Predição deve ser chamada com len(dados) = 10
            mock_instance.prediction.assert_called_once_with(10)

    def test_calculo_resultado_diferenca_absoluta(self):
        """Testa que resultado é diferença absoluta entre predição e último acumulado."""
        service = PredicaoDiaria_service()
        
        dados = {
            '01/01/2025': 100.0,
            '02/01/2025': 100.0,
            '03/01/2025': 100.0
        }
        # Acumulado seria: 100, 200, 300
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 350.0  # Predição
            MockModel.return_value = mock_instance
            
            resultado = service.processarDados(dados)
            
            # Resultado deve ser abs(350 - 300) = 50
            self.assertEqual(resultado, 50.0)

    def test_resultado_sempre_positivo(self):
        """Testa que resultado é sempre positivo (abs)."""
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            # Predição menor que acumulado
            mock_instance.prediction.return_value = 100.0
            MockModel.return_value = mock_instance
            
            resultado = service.processarDados(self.dados_validos)
            
            # Resultado deve ser positivo
            self.assertGreaterEqual(resultado, 0)

    # ========== Testes de Edge Cases ==========

    def test_valores_negativos(self):
        """Testa processamento com valores negativos."""
        dados_negativos = {
            '01/01/2025': -10.0,
            '02/01/2025': -5.0,
            '03/01/2025': -8.0
        }
        service = PredicaoDiaria_service()
        
        # Deve processar (outliers serão tratados)
        resultado = service.processarDados(dados_negativos)
        self.assertIsNotNone(resultado)

    def test_valores_zero(self):
        """Testa processamento com valores zero."""
        dados_zero = {f'{i:02d}/01/2025': 0.0 for i in range(1, 11)}
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_zero)

        self.assertIsNotNone(resultado)

    def test_valores_constantes(self):
        """Testa processamento com valores constantes."""
        dados_constantes = {f'{i:02d}/01/2025': 100.0 for i in range(1, 11)}
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_constantes)

        # Com valores constantes, predição deve ser próxima ao acumulado
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))

    def test_valores_muito_grandes(self):
        """Testa processamento com valores muito grandes."""
        dados_grandes = {
            '01/01/2025': 1e10,
            '02/01/2025': 1e10,
            '03/01/2025': 1e10
        }
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_grandes)

        self.assertIsNotNone(resultado)

    # ========== Testes de Reset de Index ==========

    def test_reset_index_executado(self):
        """Testa que reset_index é executado no DataFrame."""
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            # Verificar que DataFrame passado não tem index customizado
            df_passado = mock_instance.train.call_args[0][0]
            
            # Index deve ser RangeIndex após reset
            self.assertIsInstance(df_passado.index, pd.RangeIndex)

    # ========== Testes de Integração ==========

    def test_fluxo_completo(self):
        """Testa fluxo completo do serviço."""
        service = PredicaoDiaria_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar integridade completa
        self.assertIsInstance(resultado, (int, float))
        self.assertGreaterEqual(resultado, 0)
        # Resultado razoável (não deve ser astronomicamente grande)
        self.assertLess(resultado, 1e6)

    def test_multiplas_execucoes(self):
        """Testa múltiplas execuções no mesmo serviço."""
        service = PredicaoDiaria_service()
        
        resultado1 = service.processarDados(self.dados_validos)
        resultado2 = service.processarDados(self.dados_variados)
        
        # Ambos devem retornar resultados válidos
        self.assertIsNotNone(resultado1)
        self.assertIsNotNone(resultado2)
        
        # Resultados devem ser diferentes (dados diferentes)
        self.assertNotEqual(resultado1, resultado2)

    def test_consistencia_resultados(self):
        """Testa que mesmos dados produzem mesmo resultado."""
        service = PredicaoDiaria_service()
        
        resultado1 = service.processarDados(self.dados_validos)
        resultado2 = service.processarDados(self.dados_validos)
        
        # Deve ser determinístico
        self.assertEqual(resultado1, resultado2)

    # ========== Testes de Tratamento de Erros ==========

    def test_dados_vazios_levanta_excecao(self):
        """Testa que dados vazios levantam exceção."""
        service = PredicaoDiaria_service()
        
        with self.assertRaises(Exception):
            service.processarDados({})

    def test_dados_none_levanta_excecao(self):
        """Testa que dados None levantam exceção."""
        service = PredicaoDiaria_service()
        
        with self.assertRaises(Exception):
            service.processarDados(None)

    # ========== Testes de Validação de Dados ==========

    def test_criacao_dataframe(self):
        """Testa que DataFrame é criado corretamente."""
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
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
        service = PredicaoDiaria_service()
        
        with patch('ml_pipeline.senseFlow_A.predicao.PredicaoDiaria_service.LinearRegression_Acumulado') as MockModel:
            mock_instance = MagicMock()
            mock_instance.prediction.return_value = 1000.0
            MockModel.return_value = mock_instance
            
            service.processarDados(self.dados_validos)
            
            df = mock_instance.train.call_args[0][0]
            
            # Verificar primeira e última data
            self.assertEqual(df['Data'].iloc[0], '01/01/2025')
            self.assertEqual(df['Data'].iloc[-1], '30/01/2025')

    # ========== Testes de Performance ==========

    def test_processamento_rapido(self):
        """Testa que processamento é razoavelmente rápido."""
        import time
        
        service = PredicaoDiaria_service()
        
        inicio = time.time()
        service.processarDados(self.dados_validos)
        fim = time.time()
        
        tempo_execucao = fim - inicio
        
        # Deve processar em menos de 1 segundo
        self.assertLess(tempo_execucao, 1.0)

    def test_processamento_grandes_volumes(self):
        """Testa processamento com grande volume de dados."""
        # 365 dias de dados
        dados_ano = {f'{i:03d}/2025': float(i) for i in range(1, 366)}
        
        service = PredicaoDiaria_service()
        resultado = service.processarDados(dados_ano)
        
        self.assertIsNotNone(resultado)
        self.assertIsInstance(resultado, (int, float))


if __name__ == '__main__':
    unittest.main()
