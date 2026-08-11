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

class TokenEndpointTests(TestCase):

    def setUp(self):
        self.client = APIClient()
        self.username = 'string'
        self.password = 'string'
        get_user_model().objects.create_user(username=self.username, password=self.password)

    def test_token_obtain_pair_retorna_access_e_refresh(self):
        """Cenário: credenciais válidas são enviadas ao endpoint de token.
        Resultado esperado: a API retorna access e refresh com HTTP 200."""
        response = self.client.post(reverse('token_obtain_pair'), {'username': self.username, 'password': self.password}, format='json')
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', payload)
        self.assertIn('refresh', payload)
        self.assertIsInstance(payload['access'], str)
        self.assertIsInstance(payload['refresh'], str)

class PredictionAndAnalysisAPITests(APITestCase):

    def setUp(self):
        self.user = get_user_model().objects.create_user(username='api_tester', password='strong-password-123')
        self.client.force_authenticate(user=self.user)

    def test_v2_predicao_diaria_sucesso_retornando_prediction(self):
        """Cenário: o endpoint diário V2 recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = {'sensor_id': 'sensor_1'}
        historico = build_daily_history()
        with patch('appSM.api.views.PredicaoUseCase') as mock_uc_cls:
            mock_uc_cls.diario.return_value = 19.5
            response = self.client.post(reverse('v2-predicao-consumo-diario'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {'Prediction': 19.5})
        mock_uc_cls.diario.assert_called_once_with(sensor_id='sensor_1')

    def test_v2_predicao_mensal_sucesso_retornando_prediction(self):
        """Cenário: o endpoint mensal V2 recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = {'unidade_id': 10}
        historico = build_monthly_history()
        with patch('appSM.api.views.PredicaoUseCase') as mock_uc_cls:
            mock_uc_cls.mensal.return_value = 220.0
            response = self.client.post(reverse('v2-predicao-consumo-mensal'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {'Prediction': 220.0})
        mock_uc_cls.mensal.assert_called_once_with(unidade_id=10, dispositivo_id=None)

    def test_v2_predicao_diaria_payload_vazio_retorna_422(self):
        """Cenário: a requisição V2 chega com objeto vazio.
        Resultado esperado: HTTP 422 com erro de parâmetros inválidos."""
        response = self.client.post(reverse('v2-predicao-consumo-diario'), {}, format='json')
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error'], 'Parâmetros inválidos')

    def test_v2_predicao_mensal_json_malformado_retorna_400(self):
        """Cenário: o JSON enviado é inválido.
        Resultado esperado: HTTP 400 com mensagem de JSON mal formatado."""
        response = self.client.generic('POST', reverse('v2-predicao-consumo-mensal'), data='{invalid-json', content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'error': 'JSON mal formatado.'})

    def test_v2_predicao_diaria_erro_interno_retorna_500(self):
        """Cenário: o serviço lança uma exceção inesperada.
        Resultado esperado: HTTP 500 com mensagem genérica de erro interno."""
        payload = {'sensor_id': 'sensor_1'}
        with patch('appSM.api.views.PredicaoUseCase.diario', side_effect=Exception('falha inesperada')):
            response = self.client.post(reverse('v2-predicao-consumo-diario'), payload, format='json')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {'error': 'Erro interno.'})

    def test_v2_predicao_diaria_exige_autenticacao(self):
        """Cenário: a rota V2 é chamada sem autenticação.
        Resultado esperado: HTTP 401 antes de qualquer execução da view."""
        anon_client = APIClient()
        response = anon_client.post(reverse('v2-predicao-consumo-diario'), {'sensor_id': 'sensor_1'}, format='json')
        self.assertEqual(response.status_code, 401)

    def test_v2_classification_history_daily_sucesso(self):
        """Cenario: relatorio historico diario recebe filtros validos.
        Resultado esperado: HTTP 200 e payload com 'results' e lista processada."""
        payload = {'type': 'daily', 'unidade_id': 10, 'data_inicio': '2026-06-01', 'data_fim': '2026-06-30'}
        mock_res = {'results': [{'periodo': '2026-06-01', 'consumo': 12.0, 'classificacao': 'Crítico'}]}
        with patch('appSM.api.views.HistoricoUseCase') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = mock_res
            response = self.client.post(reverse('v2-classification-history'), payload, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_res)
        mock_service.processar.assert_called_once()
        args = mock_service.processar.call_args[0][0]
        self.assertEqual(args['type'], 'daily')
        self.assertEqual(args['unidade_id'], 10)

    def test_v2_classification_history_periodo_invalido_retorna_422(self):
        """Cenario: data_inicio e maior que data_fim.
        Resultado esperado: HTTP 422 de erro de validacao do serializer."""
        payload = {'type': 'daily', 'unidade_id': 10, 'data_inicio': '2026-06-30', 'data_fim': '2026-06-01'}
        response = self.client.post(reverse('v2-classification-history'), payload, format='json')
        self.assertEqual(response.status_code, 422)

    def test_v2_analise_diaria_sucesso_retorna_classificacao(self):
        """Cenário: a análise diária V2 recebe dados válidos.
        Resultado esperado: HTTP 200 com Data, Consumo e classificacao."""
        payload = {'sensor_id': 'sensor_1'}
        historico = build_daily_history(count=30)
        with patch('appSM.api.views.EstatisticaUseCase') as mock_uc_cls:
            mock_uc_cls.diario.return_value = {'Data': '30/01/2024', 'Consumo': 28.0, 'Classificação': 1}
            response = self.client.post(reverse('v2-classificacao-consumo-diaria'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {'Data': '30/01/2024', 'Consumo': 28.0, 'classificacao': 1})
        mock_uc_cls.diario.assert_called_once_with(sensor_id='sensor_1')

    def test_v2_analise_mensal_sucesso_retorna_classificacao(self):
        """Cenário: a análise mensal V2 recebe dados válidos.
        Resultado esperado: HTTP 200 com o mesmo contrato de saída da análise diária."""
        payload = {'unidade_id': 10}
        historico = build_monthly_history()
        with patch('appSM.api.views.EstatisticaUseCase') as mock_uc_cls:
            mock_uc_cls.mensal.return_value = {'Data': '01/12/2024', 'Consumo': 111.0, 'Classificação': 3}
            response = self.client.post(reverse('v2-classificacao-consumo-mensal'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {'Data': '01/12/2024', 'Consumo': 111.0, 'classificacao': 3})
        mock_uc_cls.mensal.assert_called_once_with(unidade_id=10, dispositivo_id=None)

    def test_v2_dados_bandas_sucesso_retorna_lista_processada(self):
        """Cenário: o endpoint de bandas V2 recebe um sensor válido.
        Resultado esperado: HTTP 200 com a chave dados contendo uma lista de registros."""
        payload = {'sensor_id': 'sensor_1'}
        historico = build_daily_history(count=6)
        with patch('appSM.api.views.EstatisticaUseCase') as mock_uc_cls:
            mock_uc_cls.dados_completos.return_value = [{'Data': '01/01/2024', 'Consumo': 10.0, 'Média Móvel': 10.0, 'Desvio Padrão': 0.0}]
            response = self.client.post(reverse('v2-dados-bandas'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('dados', body)
        self.assertIsInstance(body['dados'], list)
        self.assertEqual(body['dados'][0]['Data'], '01/01/2024')
        mock_uc_cls.dados_completos.assert_called_once_with(sensor_id='sensor_1')

    def test_classificacao_ph_sucesso_retorna_payload_do_servico(self):
        """Cenário: o endpoint de pH recebe client_id e ph_value válidos.
        Resultado esperado: HTTP 200 com o payload completo devolvido pelo serviço."""
        payload = {'client_id': 'sisar', 'ph_value': 7.2}
        with patch('appSM.api.views.PHClassificationService') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.classify.return_value = {'client_id': 'sisar', 'ph_value': 7.2, 'classification': 'adequado', 'confidence': 0.95, 'model_version': 'v1.0.0'}
            response = self.client.post(reverse('classificacao-ph'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body['classification'], 'adequado')
        self.assertEqual(body['client_id'], 'sisar')
        self.assertIsInstance(body['ph_value'], float)
        mock_service_cls.assert_called_once()
        mock_service.classify.assert_called_once_with(client_id='sisar', ph_value=7.2)

    def test_classificacao_ph_sem_client_id_retorna_400(self):
        """Cenário: o campo obrigatório client_id está ausente.
        Resultado esperado: HTTP 400 com mensagem explícita de campo faltante."""
        response = self.client.post(reverse('classificacao-ph'), {'ph_value': 7.2}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'error': 'Campo obrigatório ausente: client_id'})

    def test_classificacao_ph_sem_ph_value_retorna_400(self):
        """Cenário: o campo obrigatório ph_value está ausente.
        Resultado esperado: HTTP 400 com mensagem explícita de campo faltante."""
        response = self.client.post(reverse('classificacao-ph'), {'client_id': 'sisar'}, format='json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {'error': 'Campo obrigatório ausente: ph_value'})

    def test_classificacao_ph_tipo_invalido_retorna_422(self):
        """Cenário: ph_value não pode ser convertido para número.
        Resultado esperado: HTTP 422 com mensagem informando o tipo recebido."""
        response = self.client.post(reverse('classificacao-ph'), {'client_id': 'sisar', 'ph_value': 'alto'}, format='json')
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {'error': 'ph_value deve ser um número, recebido: alto'})

    def test_classificacao_ph_modelo_nao_encontrado_retorna_404(self):
        """Cenário: o serviço informa que o modelo do cliente não existe.
        Resultado esperado: HTTP 404 com erro e detalhe do problema."""
        payload = {'client_id': 'sisar', 'ph_value': 7.2}
        with patch('appSM.api.views.PHClassificationService') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.classify.side_effect = FileNotFoundError('arquivo ausente')
            response = self.client.post(reverse('classificacao-ph'), payload, format='json')
        body = response.json()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(body['error'], 'Modelo não encontrado para este cliente')
        self.assertIn('arquivo ausente', body['detail'])

class ClassificationRangeAPITests(APITestCase):

    def setUp(self):
        self.user = get_user_model().objects.create_user(username='api_tester2', password='strong-password-123')
        self.client.force_authenticate(user=self.user)

    def test_v2_classification_range_returns_true(self):
        execution_id = 'd7d746c8-c95f-4cb0-b004-dc4995f5ef56'
        payload = {'unidade_id': 10, 'reference_period': '2026-07-27', 'execution_id': execution_id}
        service_result = {'outside_green_range': True, 'severity': 'critical', 'classification': 2, 'classification_label': 'Consumo Excessivo', 'reference_period': '2026-07-27', 'execution_id': execution_id}
        with patch('appSM.api.views.RangeUseCase') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = service_result
            response = self.client.post(reverse('v2-classification-range'), payload, format='json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), service_result)
            mock_service.processar.assert_called_once_with(10, date(2026, 7, 27), UUID(execution_id))

    def test_v2_classification_range_returns_false(self):
        payload = {'unidade_id': 10}
        service_result = {'outside_green_range': False, 'severity': 'green', 'classification': 0, 'classification_label': 'Consumo Moderado', 'reference_period': '2026-07-27'}
        with patch('appSM.api.views.RangeUseCase') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = service_result
            response = self.client.post(reverse('v2-classification-range'), payload, format='json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), service_result)
            mock_service.processar.assert_called_once_with(10, None, None)

    def test_v2_classification_range_no_data(self):
        payload = {'unidade_id': 10}
        with patch('appSM.api.views.RangeUseCase') as mock_service_cls:
            mock_service = mock_service_cls.return_value
            from appSM.application.exceptions import ConsumoNaoEncontrado
            mock_service.processar.side_effect = ConsumoNaoEncontrado('Nenhum registro encontrado no periodo solicitado')
            response = self.client.post(reverse('v2-classification-range'), payload, format='json')
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {'error': 'Nenhum registro encontrado no periodo solicitado'})

    def test_v2_classification_range_rejects_invalid_reference_period(self):
        response = self.client.post(reverse('v2-classification-range'), {'unidade_id': 10, 'reference_period': '27/07/2026'}, format='json')
        self.assertEqual(response.status_code, 422)
        self.assertIn('reference_period', response.json()['details'])