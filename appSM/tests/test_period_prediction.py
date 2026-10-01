from datetime import date
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory, force_authenticate

from appSM.api.views import V2PredicaoPeriodo
from appSM.application.predicao_use_case import PredicaoUseCase
from appSM.application.exceptions import ConsumoNaoEncontrado


class PeriodPredictionTests(SimpleTestCase):
    @patch('appSM.application.predicao_use_case.ConsumoRepository')
    def test_forecasts_exact_duration_from_prior_history(self, repository):
        repository.return_value.buscar_contexto_predicao.return_value = pd.DataFrame(
            {'Consumo': [10.0] * 30}, index=pd.date_range('2025-12-01', periods=30)
        )
        prediction = PredicaoUseCase.periodo(10, 2, date(2025, 12, 31), date(2026, 1, 5))
        self.assertAlmostEqual(prediction, 60.0)
        repository.return_value.buscar_contexto_predicao.assert_called_once_with(10, 2, date(2025, 12, 31))

    @patch('appSM.application.predicao_use_case.ConsumoRepository')
    def test_does_not_present_insufficient_history_as_zero(self, repository):
        repository.return_value.buscar_contexto_predicao.return_value = pd.DataFrame(
            {'Consumo': [10.0]}, index=pd.date_range('2025-12-30', periods=1)
        )
        with self.assertRaises(ConsumoNaoEncontrado):
            PredicaoUseCase.periodo(10, 1, date(2025, 12, 31), date(2026, 1, 5))

    @patch('appSM.api.views.PredicaoUseCase.periodo', return_value=60.0)
    def test_endpoint_forwards_dates_and_resource(self, prediction):
        request = APIRequestFactory().post('/v2/prediction/period', {
            'unidade_id': 10, 'resourceId': 2, 'data_inicio': '2025-12-31', 'data_fim': '2026-01-05',
        }, format='json')
        force_authenticate(request, user=SimpleNamespace(is_authenticated=True))
        response = V2PredicaoPeriodo.as_view()(request)
        self.assertEqual(response.status_code, 200)
        prediction.assert_called_once_with(unidade_id=10, resourceId=2, data_inicio=date(2025, 12, 31), data_fim=date(2026, 1, 5))

    def test_endpoint_rejects_an_inverted_interval(self):
        request = APIRequestFactory().post('/v2/prediction/period', {
            'unidade_id': 10, 'resourceId': 1, 'data_inicio': '2026-01-05', 'data_fim': '2025-12-31',
        }, format='json')
        force_authenticate(request, user=SimpleNamespace(is_authenticated=True))
        self.assertEqual(V2PredicaoPeriodo.as_view()(request).status_code, 422)
