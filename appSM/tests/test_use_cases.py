from datetime import date
from unittest.mock import patch
from uuid import UUID
import pandas as pd
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import SimpleTestCase, TestCase
from rest_framework import serializers
from rest_framework.test import APIClient, APITestCase

from appSM.application.historico_use_case import HistoricoUseCase
from appSM.tests.utils import build_daily_history, build_monthly_history
from appSM.application.predicao_use_case import PredicaoUseCase
from appSM.application.estatistica_use_case import EstatisticaUseCase
from appSM.application.range_use_case import RangeUseCase
from appSM.application.exceptions import ConsumoNaoEncontrado
from appSM.infrastructure.exceptions import DataNotFoundError
from .test_characterization import *

class PredicaoUseCaseTests(SimpleTestCase):

    def test_processar_dados_validos_treina_modelo_e_retorna_float(self):
        """Cenário: histórico válido é enviado ao serviço.
        Resultado esperado: o serviço normaliza os dados, treina o modelo mockado e retorna um float."""
        with patch('appSM.application.predicao_use_case.LinearRegressionAcumulado') as mock_model_cls, patch('appSM.application.predicao_use_case.ConsumoRepository') as mock_repo_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.return_value = 12.75
            import pandas as pd
            d = build_daily_history()
            df = pd.DataFrame({'Data': pd.to_datetime(list(d.keys()), format='%d/%m/%Y'), 'Consumo': list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            resultado = PredicaoUseCase.diario('sensor_1')
            self.assertIsInstance(resultado, float)
            self.assertEqual(resultado, 12.75)
            mock_model.treinar.assert_called_once()
            mock_model.prever.assert_called_once()
            dataframe_enviado = mock_model.treinar.call_args.args[0]
            self.assertEqual(list(dataframe_enviado.columns), ['Data', 'Consumo'])
            self.assertEqual(len(dataframe_enviado), 4)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: o payload chega vazio.
        Resultado esperado: o serviço interrompe o fluxo com ValueError sem chamar o modelo."""
        with patch('appSM.application.predicao_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            mock_repo_cls.return_value.buscar_historico_diario.return_value = pd.DataFrame()
            with self.assertRaisesMessage(ValueError, 'Nenhuma data valida encontrada no historico'):
                PredicaoUseCase.diario('sensor_1')

    def test_processar_dados_com_datas_invalidas_gera_value_error(self):
        """Cenário: todas as datas recebidas são inválidas.
        Resultado esperado: a normalização falha com ValueError antes de treinar qualquer modelo."""
        with patch('appSM.application.predicao_use_case.LinearRegressionAcumulado') as mock_model_cls, patch('appSM.application.predicao_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            df = pd.DataFrame({'Data': ['31/02/2024'], 'Consumo': [10.0]})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            with self.assertRaisesMessage(ValueError, 'Nenhuma data valida encontrada no historico'):
                PredicaoUseCase.diario('sensor_1')
            mock_model_cls.return_value.treinar.assert_not_called()

    def test_processar_dados_propaga_excecao_inesperada_do_modelo(self):
        """Cenário: o modelo mockado quebra durante a predição.
        Resultado esperado: o serviço converte a falha em Exception para a camada superior."""
        with patch('appSM.application.predicao_use_case.LinearRegressionAcumulado') as mock_model_cls, patch('appSM.application.predicao_use_case.ConsumoRepository') as mock_repo_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.side_effect = RuntimeError('falha na inferencia')
            import pandas as pd
            d = build_daily_history()
            df = pd.DataFrame({'Data': pd.to_datetime(list(d.keys()), format='%d/%m/%Y'), 'Consumo': list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            with self.assertRaisesMessage(Exception, 'Erro na predição: falha na inferencia'):
                PredicaoUseCase.diario('sensor_1')

    def test_tratar_outliers_mediana_nao_altera_variacao_leve(self):
        """Cenário: a série tem apenas uma oscilação pequena.
        Resultado esperado: nenhum ponto é marcado como outlier."""
        df = pd.DataFrame({'Data': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'], 'Consumo': [100.0, 101.0, 102.0, 103.0, 107.0]})
        tratado, mascara = tratar_outliers_iqr(df.copy(), multiplicador=3.0, substituicao='mediana')
        self.assertEqual(int(mascara.sum()), 0)
        self.assertTrue(tratado['Consumo'].equals(df['Consumo']))

    def test_tratar_outliers_mediana_trata_pico_extremo(self):
        """Cenário: a série contém um pico muito acima do padrão.
        Resultado esperado: apenas o pico é substituído pela mediana."""
        df = pd.DataFrame({'Data': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'], 'Consumo': [100.0, 101.0, 102.0, 103.0, 300.0]})
        tratado, mascara = tratar_outliers_iqr(df.copy(), multiplicador=3.0, substituicao='mediana')
        self.assertEqual(int(mascara.sum()), 1)
        self.assertEqual(float(tratado.loc[mascara, 'Consumo'].iloc[0]), 101.5)

class EstatisticaUseCaseTests(SimpleTestCase):

    def test_processar_dados_validos_retorna_dict_com_estrutura_esperada(self):
        """Cenário: um histórico mensal válido é enviado à análise estatística.
        Resultado esperado: o serviço retorna um dicionário com Data, Consumo e Classificação."""
        with patch('appSM.application.estatistica_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            d = build_monthly_history()
            df = pd.DataFrame({'Data': pd.to_datetime(list(d.keys()), format='%d/%m/%Y'), 'Consumo': list(d.values())})
            mock_repo_cls.return_value.buscar_historico_mensal.return_value = df
            resultado = EstatisticaUseCase.mensal(10)
        self.assertIsInstance(resultado, dict)
        self.assertEqual(set(resultado.keys()), {'Data', 'Consumo', 'Classificação'})
        self.assertIsInstance(resultado['Data'], str)
        self.assertIsInstance(resultado['Consumo'], float)
        self.assertIsInstance(resultado['Classificação'], (int, float, str))

    def test_obter_dados_completos_retornam_lista_de_dicionarios(self):
        """Cenário: o serviço recebe um histórico válido para bandas completas.
        Resultado esperado: a resposta é uma lista de registros com colunas processadas e tipos consistentes."""
        with patch('appSM.application.estatistica_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            d = build_daily_history(count=8)
            df = pd.DataFrame({'Data': pd.to_datetime(list(d.keys()), format='%d/%m/%Y'), 'Consumo': list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            resultado = EstatisticaUseCase.dados_completos('sensor_1')
        self.assertIsInstance(resultado, list)
        self.assertGreater(len(resultado), 0)
        primeiro = resultado[0]
        self.assertIsInstance(primeiro, dict)
        self.assertIn('Data', primeiro)
        self.assertIn('Consumo', primeiro)
        self.assertIn('Média Móvel', primeiro)
        self.assertIn('Desvio Padrão', primeiro)

    def test_obter_dados_completos_preserva_consumo_original_de_outlier(self):
        """As bandas podem tratar outliers, mas a série exibida deve manter a medição real."""
        historico = {f'{day:02d}/07/2026': 18.0 if day == 15 else 4.0 + day % 2 * 0.5 for day in range(1, 31)}
        with patch('appSM.application.estatistica_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            df = pd.DataFrame({'Data': pd.to_datetime(list(historico.keys()), format='%d/%m/%Y'), 'Consumo': list(historico.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            resultado = EstatisticaUseCase.dados_completos('sensor_1')
        dia_15 = next((item for item in resultado if item['Data'] == '15/07/2026'))
        self.assertEqual(float(dia_15['Consumo']), 18.0)
        self.assertLess(float(dia_15['Média Móvel']), 18.0)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: a análise recebe um payload vazio.
        Resultado esperado: o serviço rejeita a requisição com ValueError."""
        with patch('appSM.application.estatistica_use_case.ConsumoRepository') as mock_repo_cls:
            import pandas as pd
            mock_repo_cls.return_value.buscar_historico_diario.return_value = pd.DataFrame()
            with self.assertRaisesMessage(ValueError, 'Nenhuma data valida encontrada no historico'):
                EstatisticaUseCase.diario('sensor_1')

class HistoricoUseCaseTests(SimpleTestCase):

    def test_daily_usa_contexto_e_retorna_apenas_periodo_solicitado(self):
        """Cenario: ha dados anteriores ao periodo solicitado.
        Resultado esperado: contexto entra no pipeline, mas a resposta fica restrita a janela."""

        class FakeConsumoRepo:

            def buscar_historico_relatorio_diario(self, unidade_id, data_inicio, data_fim):
                index = pd.date_range('2026-05-30', '2026-06-03', freq='D')
                return pd.DataFrame({'Consumo': [7.0, 8.0, 9.0, 10.0, 11.0]}, index=index)
        service = HistoricoUseCase(consumo_repo=FakeConsumoRepo())
        resultado = service.processar({'type': 'daily', 'unidade_id': 10, 'data_inicio': date(2026, 6, 1), 'data_fim': date(2026, 6, 3)})
        self.assertEqual(len(resultado['results']), 3)
        self.assertEqual(resultado['results'][0]['periodo'], '01/06/2026')
        self.assertEqual(resultado['results'][0]['classificacao'], 1)
        self.assertNotIn('31/05/2026', [item['periodo'] for item in resultado['results']])

    def test_periodos_mensais_respeitam_dia_fechamento(self):
        """Cenario: o fechamento configurado e dia 14.
        Resultado esperado: os ciclos do ano terminam no dia anterior do mes de referencia."""
        periodos = periodos_do_ano(2026, 14)
        self.assertEqual(periodos[0], (date(2025, 12, 14), date(2026, 1, 13)))
        self.assertEqual(periodos[1], (date(2026, 1, 14), date(2026, 2, 13)))

class RangeUseCaseTests(SimpleTestCase):

    def _test_classification_value(self, classification_value, expected_outside, expected_severity):
        with patch('appSM.application.range_use_case.HistoricoUseCase') as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {'results': [{'periodo': '27/07/2026', 'consumo': 12.0, 'classificacao': classification_value}]}
            service = RangeUseCase(historico_use_case=mock_history)
            resultado = service.processar(10, date(2026, 7, 27))
            self.assertEqual(resultado['outside_green_range'], expected_outside)
            self.assertEqual(resultado['severity'], expected_severity)
            self.assertEqual(resultado['classification'], classification_value)
            self.assertEqual(resultado['reference_period'], '2026-07-27')
            mock_history.processar.assert_called_once()
            args = mock_history.processar.call_args[0][0]
            self.assertEqual(args['type'], 'daily')
            self.assertEqual(args['unidade_id'], 10)
            self.assertEqual(args['data_inicio'], date(2026, 7, 27))
            self.assertEqual(args['data_fim'], date(2026, 7, 27))

    def test_classification_minus_2(self):
        self._test_classification_value(-2, True, 'critical')

    def test_classification_minus_1(self):
        self._test_classification_value(-1, False, 'green')

    def test_classification_0(self):
        self._test_classification_value(0, False, 'green')

    def test_classification_1(self):
        self._test_classification_value(1, True, 'warning')

    def test_classification_2(self):
        self._test_classification_value(2, True, 'critical')

    def test_no_data_raises_error(self):
        with patch('appSM.application.range_use_case.HistoricoUseCase') as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {'results': []}
            service = RangeUseCase(historico_use_case=mock_history)
            from appSM.infrastructure.exceptions import DataNotFoundError
            from appSM.application.exceptions import ConsumoNaoEncontrado
            with self.assertRaises(ConsumoNaoEncontrado):
                service.processar(10)

    def test_invalid_classification_raises_error(self):
        with patch('appSM.application.range_use_case.HistoricoUseCase') as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {'results': [{'periodo': '27/07/2026', 'consumo': 12.0, 'classificacao': 3}]}
            with self.assertRaisesRegex(ValueError, 'fora do intervalo esperado'):
                RangeUseCase(historico_use_case=mock_history).processar(10, date(2026, 7, 27))