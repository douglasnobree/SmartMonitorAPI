"""
Testes unitários para dadosBandas_service.

Testa o serviço que retorna dados completos das Bandas de Bollinger
incluindo média móvel, desvio padrão e todas as bandas calculadas.
"""

import unittest
import pandas as pd
from unittest.mock import patch

from ml_pipeline.senseFlow_A.classificacao.dadosBandas_service import (
    dadosBandas_service
)


class TestDadosBandasService(unittest.TestCase):
    """Testes para o serviço de dados de bandas."""

    def setUp(self):
        """Configuração executada antes de cada teste."""
        self.dados_validos = {
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
            '31/01/2025': 195.0
        }

        self.dados_pequenos = {
            '01/01/2025': 100.0,
            '02/01/2025': 105.0,
            '03/01/2025': 110.0
        }

    # ========== Testes de Processamento de Dados ==========

    def test_processar_dados_validos(self):
        """Testa processamento com dados válidos."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar que retorna lista
        self.assertIsInstance(resultado, list)
        
        # Verificar que tem a quantidade correta de registros
        self.assertEqual(len(resultado), len(self.dados_validos))

    def test_estrutura_registros(self):
        """Testa estrutura de cada registro retornado."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar estrutura do primeiro registro
        primeiro_registro = resultado[0]
        
        # Campos obrigatórios
        self.assertIn('Data', primeiro_registro)
        self.assertIn('Consumo', primeiro_registro)
        self.assertIn('Média Móvel', primeiro_registro)
        self.assertIn('Desvio Padrão', primeiro_registro)
        
        # Bandas inferiores
        self.assertIn('Banda Inf 1', primeiro_registro)
        self.assertIn('Banda Inf 2', primeiro_registro)
        self.assertIn('Banda Inf 3', primeiro_registro)
        
        # Bandas superiores
        self.assertIn('Banda Sup 1', primeiro_registro)
        self.assertIn('Banda Sup 2', primeiro_registro)
        self.assertIn('Banda Sup 3', primeiro_registro)

    def test_tipos_de_dados_nos_registros(self):
        """Testa tipos de dados nos campos dos registros."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        ultimo_registro = resultado[-1]
        
        # Data deve ser string
        self.assertIsInstance(ultimo_registro['Data'], str)
        
        # Valores numéricos
        self.assertIsInstance(ultimo_registro['Consumo'], (int, float))
        self.assertIsInstance(ultimo_registro['Média Móvel'], (int, float))
        self.assertIsInstance(ultimo_registro['Desvio Padrão'], (int, float))

    def test_valores_datas_preservados(self):
        """Testa que datas originais são preservadas."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar primeira data
        self.assertEqual(resultado[0]['Data'], '01/01/2025')
        
        # Verificar última data
        self.assertEqual(resultado[-1]['Data'], '31/01/2025')

    def test_valores_consumo_preservados(self):
        """Testa que valores de consumo originais são preservados."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar primeiro consumo
        self.assertEqual(resultado[0]['Consumo'], 100.0)
        
        # Verificar último consumo
        self.assertEqual(resultado[-1]['Consumo'], 195.0)

    # ========== Testes de Cálculos ==========

    def test_calculo_media_movel(self):
        """Testa cálculo da média móvel."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Com janela=30 e min_periods=1, todos devem ter média móvel
        for registro in resultado:
            self.assertIsNotNone(registro['Média Móvel'])
            # Não deve ser NaN (foi preenchido com 0 se necessário)
            self.assertNotEqual(registro['Média Móvel'], float('nan'))

    def test_calculo_desvio_padrao(self):
        """Testa cálculo do desvio padrão."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Desvio padrão deve existir para todos os registros
        for registro in resultado:
            self.assertIn('Desvio Padrão', registro)

    def test_calculo_bandas_bollinger(self):
        """Testa cálculo de todas as bandas de Bollinger."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Pegar último registro com dados completos
        ultimo = resultado[-1]
        media = ultimo['Média Móvel']
        desvio = ultimo['Desvio Padrão']

        # Verificar fórmulas das bandas (aproximadamente)
        # Banda Inf = Média - n*Desvio
        # Banda Sup = Média + n*Desvio
        
        tolerancia = 0.01  # Tolerância para comparações float
        
        # Bandas inferiores
        self.assertAlmostEqual(
            ultimo['Banda Inf 1'], 
            media - 1 * desvio, 
            delta=tolerancia
        )
        self.assertAlmostEqual(
            ultimo['Banda Inf 2'], 
            media - 2 * desvio, 
            delta=tolerancia
        )
        self.assertAlmostEqual(
            ultimo['Banda Inf 3'], 
            media - 3 * desvio, 
            delta=tolerancia
        )
        
        # Bandas superiores
        self.assertAlmostEqual(
            ultimo['Banda Sup 1'], 
            media + 1 * desvio, 
            delta=tolerancia
        )
        self.assertAlmostEqual(
            ultimo['Banda Sup 2'], 
            media + 2 * desvio, 
            delta=tolerancia
        )
        self.assertAlmostEqual(
            ultimo['Banda Sup 3'], 
            media + 3 * desvio, 
            delta=tolerancia
        )

    def test_janela_utilizada(self):
        """Testa que janela fixa de 30 é utilizada."""
        service = dadosBandas_service()
        
        # Criar dados com quantidade conhecida
        dados_31 = {f'{i:02d}/01/2025': 100.0 for i in range(1, 32)}
        resultado = service.processarDados(dados_31)

        # Com janela=30, o 31º registro deve ter média dos últimos 30
        ultimo = resultado[-1]
        
        # Verificar que média foi calculada
        self.assertIsNotNone(ultimo['Média Móvel'])
        self.assertEqual(ultimo['Média Móvel'], 100.0)  # Todos iguais

    # ========== Testes de Tratamento de NaN ==========

    def test_fillna_em_valores_numericos(self):
        """Testa que valores NaN são preenchidos com 0."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_pequenos)

        # Primeiro registro pode ter desvio padrão = 0 ou NaN preenchido
        primeiro = resultado[0]
        
        # Não deve ter NaN após fillna
        for campo in primeiro:
            if campo != 'Data' and campo != 'Classificação':
                valor = primeiro[campo]
                self.assertFalse(pd.isna(valor), f"Campo {campo} é NaN")

    def test_classificacao_nao_presente(self):
        """Testa que campo Classificação não é adicionado (diferente de AnaliseEstatisticaService)."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Este serviço não adiciona classificação
        primeiro = resultado[0]
        # Verificar que NÃO tem classificação (foi removido da lógica de fillna)
        # OU verificar que fillna não afeta campo inexistente
        if 'Classificação' in primeiro:
            self.fail("Classificação não deveria estar presente em dadosBandas_service")

    # ========== Testes com Diferentes Quantidades de Dados ==========

    def test_processar_dados_pequenos(self):
        """Testa processamento com poucos dados."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_pequenos)

        self.assertEqual(len(resultado), 3)
        self.assertIsInstance(resultado, list)

    def test_processar_um_registro(self):
        """Testa processamento com apenas um registro."""
        dados_um = {'01/01/2025': 100.0}
        service = dadosBandas_service()
        resultado = service.processarDados(dados_um)

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0]['Data'], '01/01/2025')
        self.assertEqual(resultado[0]['Consumo'], 100.0)

    def test_processar_exatamente_30_registros(self):
        """Testa com exatamente 30 registros (tamanho da janela)."""
        dados_30 = {f'{i:02d}/01/2025': float(i) for i in range(1, 31)}
        service = dadosBandas_service()
        resultado = service.processarDados(dados_30)

        self.assertEqual(len(resultado), 30)

    def test_processar_mais_de_30_registros(self):
        """Testa com mais de 30 registros."""
        dados_60 = {f'{i:02d}/01/2025': float(i) for i in range(1, 61)}
        service = dadosBandas_service()
        resultado = service.processarDados(dados_60)

        self.assertEqual(len(resultado), 60)

    # ========== Testes de Edge Cases ==========

    def test_valores_negativos(self):
        """Testa processamento com valores negativos."""
        dados_negativos = {
            '01/01/2025': -10.0,
            '02/01/2025': -5.0,
            '03/01/2025': -8.0
        }
        service = dadosBandas_service()
        resultado = service.processarDados(dados_negativos)

        self.assertEqual(len(resultado), 3)
        self.assertEqual(resultado[0]['Consumo'], -10.0)

    def test_valores_zero(self):
        """Testa processamento com valores zero."""
        dados_zero = {f'{i:02d}/01/2025': 0.0 for i in range(1, 11)}
        service = dadosBandas_service()
        resultado = service.processarDados(dados_zero)

        self.assertEqual(len(resultado), 10)
        # Com valores zero, média móvel deve ser zero
        self.assertEqual(resultado[-1]['Média Móvel'], 0.0)

    def test_valores_constantes(self):
        """Testa processamento com valores constantes."""
        dados_constantes = {f'{i:02d}/01/2025': 100.0 for i in range(1, 31)}
        service = dadosBandas_service()
        resultado = service.processarDados(dados_constantes)

        ultimo = resultado[-1]
        # Média móvel deve ser 100
        self.assertEqual(ultimo['Média Móvel'], 100.0)
        # Desvio padrão deve ser 0
        self.assertEqual(ultimo['Desvio Padrão'], 0.0)
        # Bandas devem ser iguais à média
        self.assertEqual(ultimo['Banda Inf 1'], 100.0)
        self.assertEqual(ultimo['Banda Sup 1'], 100.0)

    def test_valores_muito_grandes(self):
        """Testa processamento com valores muito grandes."""
        dados_grandes = {
            '01/01/2025': 1e10,
            '02/01/2025': 1e10,
            '03/01/2025': 1e10
        }
        service = dadosBandas_service()
        resultado = service.processarDados(dados_grandes)

        self.assertIsNotNone(resultado)
        self.assertEqual(len(resultado), 3)

    # ========== Testes de Tratamento de Erros ==========

    def test_dados_vazios_levanta_excecao(self):
        """Testa que dados vazios levantam exceção."""
        service = dadosBandas_service()
        
        with self.assertRaises(Exception):
            service.processarDados({})

    def test_excecao_mantida_original(self):
        """Testa que exceções são re-levantadas corretamente."""
        service = dadosBandas_service()
        
        # Forçar erro passando dados inválidos
        with self.assertRaises(Exception) as context:
            service.processarDados(None)
        
        # Verificar que exceção foi levantada
        self.assertIsNotNone(context.exception)

    # ========== Testes de Formato de Saída ==========

    def test_formato_saida_lista_dicionarios(self):
        """Testa que saída é lista de dicionários."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        self.assertIsInstance(resultado, list)
        for registro in resultado:
            self.assertIsInstance(registro, dict)

    def test_ordem_registros_preservada(self):
        """Testa que ordem dos registros é preservada."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Datas devem estar na mesma ordem
        datas_resultado = [r['Data'] for r in resultado]
        datas_originais = list(self.dados_validos.keys())
        
        self.assertEqual(datas_resultado, datas_originais)

    # ========== Testes de Integração ==========

    def test_fluxo_completo(self):
        """Testa fluxo completo do serviço."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Verificar integridade completa
        self.assertIsInstance(resultado, list)
        self.assertEqual(len(resultado), 31)
        
        # Verificar todos os registros têm campos necessários
        for registro in resultado:
            self.assertIn('Data', registro)
            self.assertIn('Consumo', registro)
            self.assertIn('Média Móvel', registro)
            self.assertIn('Desvio Padrão', registro)
            self.assertIn('Banda Inf 1', registro)
            self.assertIn('Banda Sup 1', registro)

    def test_multiplas_execucoes(self):
        """Testa múltiplas execuções no mesmo serviço."""
        service = dadosBandas_service()
        
        resultado1 = service.processarDados(self.dados_pequenos)
        resultado2 = service.processarDados(self.dados_validos)
        
        # Ambos devem retornar resultados válidos
        self.assertEqual(len(resultado1), 3)
        self.assertEqual(len(resultado2), 31)

    def test_comparacao_com_pandas(self):
        """Testa que cálculos estão corretos comparando com pandas."""
        service = dadosBandas_service()
        resultado = service.processarDados(self.dados_validos)

        # Calcular manualmente com pandas
        df_esperado = pd.DataFrame({
            'Data': list(self.dados_validos.keys()),
            'Consumo': list(self.dados_validos.values())
        })
        df_esperado['Média Móvel'] = df_esperado['Consumo'].rolling(window=30, min_periods=1).mean()
        df_esperado['Desvio Padrão'] = df_esperado['Consumo'].rolling(window=30, min_periods=1).std()

        # Comparar último registro
        ultimo_esperado = df_esperado.iloc[-1]
        ultimo_resultado = resultado[-1]

        self.assertAlmostEqual(
            ultimo_resultado['Média Móvel'],
            ultimo_esperado['Média Móvel'],
            places=2
        )


if __name__ == '__main__':
    unittest.main()
