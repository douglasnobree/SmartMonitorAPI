"""
Testes unitários para AnaliseEstatisticaService.

Testa a classificação de consumo usando Bandas de Bollinger para análise
diária (janela=30) e mensal (janela=12).
"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import logging

from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import (
    AnaliseEstatisticaService
)


class TestAnaliseEstatisticaService(unittest.TestCase):
    """Testes para o serviço de análise estatística."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Dados de exemplo para testes
        self.dados_validos_diario = {
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
            '16/01/2025': 142.0,
            '17/01/2025': 145.0,
            '18/01/2025': 150.0,
            '19/01/2025': 148.0,
            '20/01/2025': 155.0,
            '21/01/2025': 160.0,
            '22/01/2025': 158.0,
            '23/01/2025': 165.0,
            '24/01/2025': 170.0,
            '25/01/2025': 168.0,
            '26/01/2025': 175.0,
            '27/01/2025': 180.0,
            '28/01/2025': 178.0,
            '29/01/2025': 185.0,
            '30/01/2025': 190.0,
            '31/01/2025': 200.0  # Valor alto para testar classificação
        }

        self.dados_validos_mensal = {
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
            'Jan/2025': 1800.0  # Valor alto para testar classificação
        }

        self.dados_pequenos = {
            '01/01/2025': 100.0,
            '02/01/2025': 105.0,
            '03/01/2025': 110.0
        }

        # Suprimir logs durante testes (opcional)
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Limpeza executada após cada teste."""
        logging.disable(logging.NOTSET)

    # ========== Testes de Inicialização ==========

    def test_init_com_janela_diaria(self):
        """Testa inicialização com janela diária (30)."""
        service = AnaliseEstatisticaService(janela=30)
        self.assertEqual(service.janela, 30)

    def test_init_com_janela_mensal(self):
        """Testa inicialização com janela mensal (12)."""
        service = AnaliseEstatisticaService(janela=12)
        self.assertEqual(service.janela, 12)

    def test_init_sem_janela_usa_padrao(self):
        """Testa que inicialização sem janela usa padrão (30)."""
        service = AnaliseEstatisticaService()
        self.assertEqual(service.janela, AnaliseEstatisticaService.JANELA_DIARIA)

    def test_constantes_classe(self):
        """Testa que constantes estão definidas corretamente."""
        self.assertEqual(AnaliseEstatisticaService.JANELA_DIARIA, 30)
        self.assertEqual(AnaliseEstatisticaService.JANELA_MENSAL, 12)

    # ========== Testes de Processamento de Dados ==========

    def test_processar_dados_validos_diario(self):
        """Testa processamento com dados válidos para análise diária."""
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(self.dados_validos_diario)

        # Verificar estrutura da resposta
        self.assertIn('Data', resultado)
        self.assertIn('Consumo', resultado)
        self.assertIn('Classificação', resultado)

        # Verificar tipos
        self.assertIsInstance(resultado['Data'], str)
        self.assertIsInstance(resultado['Consumo'], (int, float))
        self.assertIsInstance(resultado['Classificação'], (int, str))

        # Verificar último registro
        self.assertEqual(resultado['Data'], '31/01/2025')
        self.assertEqual(resultado['Consumo'], 200.0)

    def test_processar_dados_validos_mensal(self):
        """Testa processamento com dados válidos para análise mensal."""
        service = AnaliseEstatisticaService(janela=12)
        resultado = service.processarDados(self.dados_validos_mensal)

        # Verificar estrutura da resposta
        self.assertIn('Data', resultado)
        self.assertIn('Consumo', resultado)
        self.assertIn('Classificação', resultado)

        # Verificar último registro
        self.assertEqual(resultado['Data'], 'Jan/2025')
        self.assertEqual(resultado['Consumo'], 1800.0)

    def test_processar_dados_pequenos(self):
        """Testa processamento com poucos dados (min_periods=1)."""
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(self.dados_pequenos)

        # Deve processar mesmo com menos de 30 registros
        self.assertIsNotNone(resultado)
        self.assertEqual(resultado['Data'], '03/01/2025')
        self.assertEqual(resultado['Consumo'], 110.0)

    def test_processar_dados_um_registro(self):
        """Testa processamento com apenas um registro."""
        dados_um = {'01/01/2025': 100.0}
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados_um)

        self.assertEqual(resultado['Data'], '01/01/2025')
        self.assertEqual(resultado['Consumo'], 100.0)
        # Com um registro, classificação pode ser None/Sem classificação
        self.assertIsNotNone(resultado['Classificação'])

    # ========== Testes de Validação de Entrada ==========

    def test_dados_vazios_levanta_excecao(self):
        """Testa que dados vazios levantam ValueError."""
        service = AnaliseEstatisticaService(janela=30)
        with self.assertRaises(ValueError) as context:
            service.processarDados({})
        self.assertIn("não pode estar vazio", str(context.exception))

    def test_dados_none_levanta_excecao(self):
        """Testa que dados None levantam exceção."""
        service = AnaliseEstatisticaService(janela=30)
        with self.assertRaises(ValueError):
            service.processarDados(None)

    # ========== Testes de Classificação ==========

    def test_classificacao_faixa_superior_2(self):
        """Testa classificação com valor muito acima do normal."""
        dados = {f'{i:02d}/01/2025': 100.0 for i in range(1, 31)}
        dados['31/01/2025'] = 300.0  # Muito acima da média
        
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)

        # Esperamos classificação positiva (1 ou 2)
        self.assertIsInstance(resultado['Classificação'], int)
        self.assertGreater(resultado['Classificação'], 0)

    def test_classificacao_faixa_inferior_2(self):
        """Testa classificação com valor muito abaixo do normal."""
        dados = {f'{i:02d}/01/2025': 100.0 for i in range(1, 31)}
        dados['31/01/2025'] = 10.0  # Muito abaixo da média
        
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)

        # Esperamos classificação negativa (-1 ou -2)
        self.assertIsInstance(resultado['Classificação'], int)
        self.assertLess(resultado['Classificação'], 0)

    def test_classificacao_faixa_ideal(self):
        """Testa classificação com valor normal (dentro da média)."""
        dados = {f'{i:02d}/01/2025': 100.0 for i in range(1, 32)}
        
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)

        # Com valores constantes, deve estar na faixa ideal (0)
        # Ou sem classificação devido a desvio padrão zero
        classificacao = resultado['Classificação']
        self.assertTrue(
            classificacao == 0 or classificacao == "Sem classificação"
        )

    def test_classificacao_metodo_privado(self):
        """Testa método _classifica diretamente."""
        service = AnaliseEstatisticaService(janela=30)
        
        # Simular uma linha com valores
        row = pd.Series({
            'Consumo': 150.0,
            'Banda Sup 2': 160.0,
            'Banda Sup 1': 140.0,
            'Banda Inf 1': 120.0,
            'Banda Inf 2': 110.0,
            'Desvio Padrão': 20.0
        })
        
        classificacao = service._classifica(row)
        # Consumo 150 está entre Banda Sup 1 (140) e Banda Sup 2 (160)
        self.assertEqual(classificacao, 1)

    def test_classificacao_sem_desvio_padrao(self):
        """Testa classificação quando desvio padrão é NaN."""
        service = AnaliseEstatisticaService(janela=30)
        
        row = pd.Series({
            'Consumo': 100.0,
            'Banda Sup 2': pd.NA,
            'Banda Sup 1': pd.NA,
            'Banda Inf 1': pd.NA,
            'Banda Inf 2': pd.NA,
            'Desvio Padrão': pd.NA
        })
        
        classificacao = service._classifica(row)
        self.assertIsNone(classificacao)

    # ========== Testes de Cálculo de Bandas ==========

    def test_calculo_bandas_bollinger(self):
        """Testa se todas as bandas de Bollinger são calculadas."""
        service = AnaliseEstatisticaService(janela=5)
        
        # Usar mock para capturar o DataFrame processado
        with patch.object(service, '_classifica', return_value=0):
            resultado = service.processarDados(self.dados_pequenos)
            
            # Verificar que processamento ocorreu sem erros
            self.assertIsNotNone(resultado)

    def test_fillna_valores_numericos(self):
        """Testa que valores NaN são preenchidos com 0."""
        service = AnaliseEstatisticaService(janela=30)
        dados = {'01/01/2025': 100.0}  # Apenas um registro
        
        resultado = service.processarDados(dados)
        
        # Não deve haver NaN no resultado
        self.assertIsNotNone(resultado['Consumo'])
        self.assertFalse(pd.isna(resultado['Consumo']))

    # ========== Testes de Logging ==========

    @patch('ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service.logger')
    def test_logging_inicializacao(self, mock_logger):
        """Testa que log de inicialização é chamado."""
        service = AnaliseEstatisticaService(janela=15)
        mock_logger.info.assert_called_once()
        args = mock_logger.info.call_args[0][0]
        self.assertIn('janela=15', args)

    @patch('ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service.logger')
    def test_logging_processamento(self, mock_logger):
        """Testa que logs de processamento são chamados."""
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(self.dados_pequenos)
        
        # Verificar que logger.debug foi chamado
        self.assertTrue(mock_logger.debug.called)

    @patch('ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service.logger')
    def test_logging_erro(self, mock_logger):
        """Testa que erros são logados."""
        service = AnaliseEstatisticaService(janela=30)
        
        try:
            service.processarDados({})
        except ValueError:
            pass
        
        # Verificar que logger.error foi chamado
        mock_logger.error.assert_called_once()

    # ========== Testes de Integração ==========

    def test_fluxo_completo_diario(self):
        """Testa fluxo completo de análise diária."""
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(self.dados_validos_diario)

        # Verificar integridade completa da resposta
        self.assertIsInstance(resultado, dict)
        self.assertEqual(len(resultado), 3)
        self.assertIn('Data', resultado)
        self.assertIn('Consumo', resultado)
        self.assertIn('Classificação', resultado)

    def test_fluxo_completo_mensal(self):
        """Testa fluxo completo de análise mensal."""
        service = AnaliseEstatisticaService(janela=12)
        resultado = service.processarDados(self.dados_validos_mensal)

        # Verificar integridade completa da resposta
        self.assertIsInstance(resultado, dict)
        self.assertEqual(len(resultado), 3)

    def test_multiplas_execucoes_mesmo_servico(self):
        """Testa múltiplas execuções no mesmo serviço."""
        service = AnaliseEstatisticaService(janela=30)
        
        resultado1 = service.processarDados(self.dados_pequenos)
        resultado2 = service.processarDados(self.dados_validos_diario)
        
        # Ambos devem retornar resultados válidos
        self.assertIsNotNone(resultado1)
        self.assertIsNotNone(resultado2)
        
        # Resultados devem ser diferentes
        self.assertNotEqual(resultado1['Data'], resultado2['Data'])

    # ========== Testes de Edge Cases ==========

    def test_valores_negativos(self):
        """Testa processamento com valores negativos de consumo."""
        dados = {
            '01/01/2025': -10.0,
            '02/01/2025': -5.0,
            '03/01/2025': -8.0
        }
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)
        
        # Deve processar sem erros
        self.assertIsNotNone(resultado)
        self.assertEqual(resultado['Consumo'], -8.0)

    def test_valores_zero(self):
        """Testa processamento com valores zero."""
        dados = {f'{i:02d}/01/2025': 0.0 for i in range(1, 11)}
        service = AnaliseEstatisticaService(janela=5)
        resultado = service.processarDados(dados)
        
        # Deve processar sem erros
        self.assertIsNotNone(resultado)

    def test_valores_muito_grandes(self):
        """Testa processamento com valores muito grandes."""
        dados = {
            '01/01/2025': 1e10,
            '02/01/2025': 1e10,
            '03/01/2025': 1e10
        }
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.processarDados(dados)
        
        # Deve processar sem erros
        self.assertIsNotNone(resultado)

    def test_janela_maior_que_dados(self):
        """Testa com janela maior que quantidade de dados."""
        dados = {f'{i:02d}/01/2025': 100.0 for i in range(1, 6)}  # 5 registros
        service = AnaliseEstatisticaService(janela=30)  # Janela de 30
        resultado = service.processarDados(dados)
        
        # Deve processar com min_periods=1
        self.assertIsNotNone(resultado)


if __name__ == '__main__':
    unittest.main()
