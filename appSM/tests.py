from datetime import date
from unittest.mock import patch
from uuid import UUID

import pandas as pd

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import SimpleTestCase, TestCase

from rest_framework import serializers
from rest_framework.test import APIClient, APITestCase

from appSM.api.serializers import MySerializer
from appSM.application.historico_use_case import HistoricoUseCase
from appSM.application.predicao_use_case import PredicaoUseCase
from appSM.application.estatistica_use_case import EstatisticaUseCase
from appSM.application.range_use_case import RangeUseCase
from appSM.application.exceptions import ConsumoNaoEncontrado
from appSM.infrastructure.exceptions import DataNotFoundError


def build_daily_history(count=5, start_year=2024, start_month=1, start_day=1, base_value=10.0):
    payload = {}
    current_date = date(start_year, start_month, start_day)

    for index in range(count):
        payload[current_date.strftime("%d/%m/%Y")] = float(base_value + index)
        current_date = date.fromordinal(current_date.toordinal() + 1)
    
    return payload


def build_monthly_history(count=12, start_year=2024, start_month=1, start_day=1, base_value=100.0):
    payload = {}
    current_year = start_year
    current_month = start_month

    for index in range(count):
        payload[date(current_year, current_month, start_day).strftime("%d/%m/%Y")] = float(base_value + index)
        total_months = current_year * 12 + (current_month - 1) + 1
        current_year = total_months // 12
        current_month = total_months % 12 + 1

    return payload


class MySerializerTests(SimpleTestCase):
    def test_rejeita_payload_nao_dict(self):
        """Cenário: a validação recebe um tipo inválido.
        Resultado esperado: o serializer rejeita a entrada com ValidationError."""
        serializer = MySerializer()

        with self.assertRaises(serializers.ValidationError) as captured_exception:
            serializer.to_internal_value([("01/01/2024", 10.0)])

        self.assertIn("Os dados devem ser um dicionário.", str(captured_exception.exception))

    def test_rejeita_data_e_valor_invalidos(self):
        """Cenário: a chave não segue DD/MM/YYYY e o valor não é numérico.
        Resultado esperado: a validação falha com mensagem explícita do campo inválido."""
        serializer = MySerializer()

        with self.assertRaises(serializers.ValidationError) as captured_exception:
            serializer.to_internal_value({"2024-01-01": "dez"})

        self.assertIn("A chave '2024-01-01' não está no formato DD/MM/YYYY.", str(captured_exception.exception))


class PredicaoUseCaseTests(SimpleTestCase):
    def test_processar_dados_validos_treina_modelo_e_retorna_float(self):
        """Cenário: histórico válido é enviado ao serviço.
        Resultado esperado: o serviço normaliza os dados, treina o modelo mockado e retorna um float."""
        with patch("appSM.application.predicao_use_case.LinearRegressionAcumulado") as mock_model_cls, \
             patch("appSM.application.predicao_use_case.ConsumoRepository") as mock_repo_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.return_value = 12.75
            import pandas as pd
            d = build_daily_history()
            df = pd.DataFrame({"Data": pd.to_datetime(list(d.keys()), format="%d/%m/%Y"), "Consumo": list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df

            resultado = PredicaoUseCase.diario("sensor_1")

            self.assertIsInstance(resultado, float)
            self.assertEqual(resultado, 12.75)
            mock_model.treinar.assert_called_once()
            mock_model.prever.assert_called_once()

            dataframe_enviado = mock_model.treinar.call_args.args[0]
            self.assertEqual(list(dataframe_enviado.columns), ["Data", "Consumo"])
            self.assertEqual(len(dataframe_enviado), 4)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: o payload chega vazio.
        Resultado esperado: o serviço interrompe o fluxo com ValueError sem chamar o modelo."""
        with patch("appSM.application.predicao_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            mock_repo_cls.return_value.buscar_historico_diario.return_value = pd.DataFrame()
            with self.assertRaisesMessage(ValueError, "Nenhuma data valida encontrada no historico"):
                PredicaoUseCase.diario("sensor_1")

    def test_processar_dados_com_datas_invalidas_gera_value_error(self):
        """Cenário: todas as datas recebidas são inválidas.
        Resultado esperado: a normalização falha com ValueError antes de treinar qualquer modelo."""
        with patch("appSM.application.predicao_use_case.LinearRegressionAcumulado") as mock_model_cls, \
             patch("appSM.application.predicao_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            df = pd.DataFrame({"Data": ["31/02/2024"], "Consumo": [10.0]})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df

            with self.assertRaisesMessage(ValueError, "Nenhuma data valida encontrada no historico"):
                PredicaoUseCase.diario("sensor_1")

            mock_model_cls.return_value.treinar.assert_not_called()

    def test_processar_dados_propaga_excecao_inesperada_do_modelo(self):
        """Cenário: o modelo mockado quebra durante a predição.
        Resultado esperado: o serviço converte a falha em Exception para a camada superior."""
        with patch("appSM.application.predicao_use_case.LinearRegressionAcumulado") as mock_model_cls, \
             patch("appSM.application.predicao_use_case.ConsumoRepository") as mock_repo_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.side_effect = RuntimeError("falha na inferencia")
            import pandas as pd
            d = build_daily_history()
            df = pd.DataFrame({"Data": pd.to_datetime(list(d.keys()), format="%d/%m/%Y"), "Consumo": list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df

            with self.assertRaisesMessage(Exception, "Erro na predição: falha na inferencia"):
                PredicaoUseCase.diario("sensor_1")

    def test_tratar_outliers_mediana_nao_altera_variacao_leve(self):
        """Cenário: a série tem apenas uma oscilação pequena.
        Resultado esperado: nenhum ponto é marcado como outlier."""
        df = pd.DataFrame(
            {
                "Data": [
                    "01/01/2024",
                    "02/01/2024",
                    "03/01/2024",
                    "04/01/2024",
                    "05/01/2024",
                ],
                "Consumo": [100.0, 101.0, 102.0, 103.0, 107.0],
            }
        )

        tratado, mascara = tratar_outliers_iqr(df.copy(), multiplicador=3.0, substituicao="mediana")

        self.assertEqual(int(mascara.sum()), 0)
        self.assertTrue(tratado["Consumo"].equals(df["Consumo"]))

    def test_tratar_outliers_mediana_trata_pico_extremo(self):
        """Cenário: a série contém um pico muito acima do padrão.
        Resultado esperado: apenas o pico é substituído pela mediana."""
        df = pd.DataFrame(
            {
                "Data": [
                    "01/01/2024",
                    "02/01/2024",
                    "03/01/2024",
                    "04/01/2024",
                    "05/01/2024",
                ],
                "Consumo": [100.0, 101.0, 102.0, 103.0, 300.0],
            }
        )

        tratado, mascara = tratar_outliers_iqr(df.copy(), multiplicador=3.0, substituicao="mediana")

        self.assertEqual(int(mascara.sum()), 1)
        self.assertEqual(float(tratado.loc[mascara, "Consumo"].iloc[0]), 101.5)


class EstatisticaUseCaseTests(SimpleTestCase):
    def test_processar_dados_validos_retorna_dict_com_estrutura_esperada(self):
        """Cenário: um histórico mensal válido é enviado à análise estatística.
        Resultado esperado: o serviço retorna um dicionário com Data, Consumo e Classificação."""
        with patch("appSM.application.estatistica_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            d = build_monthly_history()
            df = pd.DataFrame({"Data": pd.to_datetime(list(d.keys()), format="%d/%m/%Y"), "Consumo": list(d.values())})
            mock_repo_cls.return_value.buscar_historico_mensal.return_value = df
            resultado = EstatisticaUseCase.mensal(10)

        self.assertIsInstance(resultado, dict)
        self.assertEqual(set(resultado.keys()), {"Data", "Consumo", "Classificação"})
        self.assertIsInstance(resultado["Data"], str)
        self.assertIsInstance(resultado["Consumo"], float)
        self.assertIsInstance(resultado["Classificação"], (int, float, str))

    def test_obter_dados_completos_retornam_lista_de_dicionarios(self):
        """Cenário: o serviço recebe um histórico válido para bandas completas.
        Resultado esperado: a resposta é uma lista de registros com colunas processadas e tipos consistentes."""
        with patch("appSM.application.estatistica_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            d = build_daily_history(count=8)
            df = pd.DataFrame({"Data": pd.to_datetime(list(d.keys()), format="%d/%m/%Y"), "Consumo": list(d.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            resultado = EstatisticaUseCase.dados_completos("sensor_1")

        self.assertIsInstance(resultado, list)
        self.assertGreater(len(resultado), 0)
        primeiro = resultado[0]
        self.assertIsInstance(primeiro, dict)
        self.assertIn("Data", primeiro)
        self.assertIn("Consumo", primeiro)
        self.assertIn("Média Móvel", primeiro)
        self.assertIn("Desvio Padrão", primeiro)

    def test_obter_dados_completos_preserva_consumo_original_de_outlier(self):
        """As bandas podem tratar outliers, mas a série exibida deve manter a medição real."""
        historico = {
            f"{day:02d}/07/2026": 18.0 if day == 15 else 4.0 + (day % 2) * 0.5
            for day in range(1, 31)
        }

        with patch("appSM.application.estatistica_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            df = pd.DataFrame({"Data": pd.to_datetime(list(historico.keys()), format="%d/%m/%Y"), "Consumo": list(historico.values())})
            mock_repo_cls.return_value.buscar_historico_diario.return_value = df
            resultado = EstatisticaUseCase.dados_completos("sensor_1")
        dia_15 = next(item for item in resultado if item["Data"] == "15/07/2026")

        self.assertEqual(float(dia_15["Consumo"]), 18.0)
        self.assertLess(float(dia_15["Média Móvel"]), 18.0)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: a análise recebe um payload vazio.
        Resultado esperado: o serviço rejeita a requisição com ValueError."""
        with patch("appSM.application.estatistica_use_case.ConsumoRepository") as mock_repo_cls:
            import pandas as pd
            mock_repo_cls.return_value.buscar_historico_diario.return_value = pd.DataFrame()
            with self.assertRaisesMessage(ValueError, "Nenhuma data valida encontrada no historico"):
                EstatisticaUseCase.diario("sensor_1")


class HistoricoUseCaseTests(SimpleTestCase):
    def test_daily_usa_contexto_e_retorna_apenas_periodo_solicitado(self):
        """Cenario: ha dados anteriores ao periodo solicitado.
        Resultado esperado: contexto entra no pipeline, mas a resposta fica restrita a janela."""

        class FakeConsumoRepo:
            def buscar_historico_relatorio_diario(self, unidade_id, data_inicio, data_fim):
                index = pd.date_range("2026-05-30", "2026-06-03", freq="D")
                return pd.DataFrame({"Consumo": [7.0, 8.0, 9.0, 10.0, 11.0]}, index=index)

        service = HistoricoUseCase(consumo_repo=FakeConsumoRepo())

        resultado = service.processar(
            {
                "type": "daily",
                "unidade_id": 10,
                "data_inicio": date(2026, 6, 1),
                "data_fim": date(2026, 6, 3),
            }
        )

        self.assertEqual(len(resultado["results"]), 3)
        self.assertEqual(resultado["results"][0]["periodo"], "01/06/2026")
        self.assertEqual(resultado["results"][0]["classificacao"], 1)

        self.assertNotIn("31/05/2026", [item["periodo"] for item in resultado["results"]])

    def test_periodos_mensais_respeitam_dia_fechamento(self):
        """Cenario: o fechamento configurado e dia 14.
        Resultado esperado: os ciclos do ano terminam no dia anterior do mes de referencia."""
        periodos = periodos_do_ano(2026, 14)

        self.assertEqual(periodos[0], (date(2025, 12, 14), date(2026, 1, 13)))
        self.assertEqual(periodos[1], (date(2026, 1, 14), date(2026, 2, 13)))


class TokenEndpointTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.username = "string"
        self.password = "string"
        get_user_model().objects.create_user(
            username=self.username,
            password=self.password,
        )

    def test_token_obtain_pair_retorna_access_e_refresh(self):
        """Cenário: credenciais válidas são enviadas ao endpoint de token.
        Resultado esperado: a API retorna access e refresh com HTTP 200."""
        response = self.client.post(
            reverse("token_obtain_pair"),
            {"username": self.username, "password": self.password},
            format="json",
        )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", payload)
        self.assertIn("refresh", payload)
        self.assertIsInstance(payload["access"], str)
        self.assertIsInstance(payload["refresh"], str)


class PredictionAndAnalysisAPITests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="api_tester",
            password="strong-password-123",
        )
        self.client.force_authenticate(user=self.user)

    def test_v2_predicao_diaria_sucesso_retornando_prediction(self):
        """Cenário: o endpoint diário V2 recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = {"sensor_id": "sensor_1"}
        historico = build_daily_history()

        with patch("appSM.api.views.PredicaoUseCase") as mock_uc_cls:
            mock_uc_cls.diario.return_value = 19.5

            response = self.client.post(reverse("v2-predicao-consumo-diario"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Prediction": 19.5})
        mock_uc_cls.diario.assert_called_once_with(sensor_id="sensor_1")

    def test_v2_predicao_mensal_sucesso_retornando_prediction(self):
        """Cenário: o endpoint mensal V2 recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = {"unidade_id": 10}
        historico = build_monthly_history()

        with patch("appSM.api.views.PredicaoUseCase") as mock_uc_cls:
            mock_uc_cls.mensal.return_value = 220.0

            response = self.client.post(reverse("v2-predicao-consumo-mensal"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Prediction": 220.0})
        mock_uc_cls.mensal.assert_called_once_with(unidade_id=10, dispositivo_id=None)

    def test_v2_predicao_diaria_payload_vazio_retorna_422(self):
        """Cenário: a requisição V2 chega com objeto vazio.
        Resultado esperado: HTTP 422 com erro de parâmetros inválidos."""
        response = self.client.post(reverse("v2-predicao-consumo-diario"), {}, format="json")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "Parâmetros inválidos")

    def test_v2_predicao_mensal_json_malformado_retorna_400(self):
        """Cenário: o JSON enviado é inválido.
        Resultado esperado: HTTP 400 com mensagem de JSON mal formatado."""
        response = self.client.generic(
            "POST",
            reverse("v2-predicao-consumo-mensal"),
            data="{invalid-json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "JSON mal formatado."})

    def test_v2_predicao_diaria_erro_interno_retorna_500(self):
        """Cenário: o serviço lança uma exceção inesperada.
        Resultado esperado: HTTP 500 com mensagem genérica de erro interno."""
        payload = {"sensor_id": "sensor_1"}
        with patch("appSM.api.views.PredicaoUseCase.diario", side_effect=Exception("falha inesperada")):
            response = self.client.post(reverse("v2-predicao-consumo-diario"), payload, format="json")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Erro interno."})

    def test_v2_predicao_diaria_exige_autenticacao(self):
        """Cenário: a rota V2 é chamada sem autenticação.
        Resultado esperado: HTTP 401 antes de qualquer execução da view."""
        anon_client = APIClient()
        response = anon_client.post(reverse("v2-predicao-consumo-diario"), {"sensor_id": "sensor_1"}, format="json")
        self.assertEqual(response.status_code, 401)

    def test_v2_classification_history_daily_sucesso(self):
        """Cenario: relatorio historico diario recebe filtros validos.
        Resultado esperado: HTTP 200 e payload com 'results' e lista processada."""
        payload = {
            "type": "daily",
            "unidade_id": 10,
            "data_inicio": "2026-06-01",
            "data_fim": "2026-06-30",
        }
        mock_res = {"results": [{"periodo": "2026-06-01", "consumo": 12.0, "classificacao": "Crítico"}]}

        with patch("appSM.api.views.HistoricoUseCase") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = mock_res
            response = self.client.post(reverse("v2-classification-history"), payload, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_res)
        mock_service.processar.assert_called_once()
        args = mock_service.processar.call_args[0][0]
        self.assertEqual(args["type"], "daily")
        self.assertEqual(args["unidade_id"], 10)

    def test_v2_classification_history_periodo_invalido_retorna_422(self):
        """Cenario: data_inicio e maior que data_fim.
        Resultado esperado: HTTP 422 de erro de validacao do serializer."""
        payload = {
            "type": "daily",
            "unidade_id": 10,
            "data_inicio": "2026-06-30",
            "data_fim": "2026-06-01",
        }

        response = self.client.post(reverse("v2-classification-history"), payload, format="json")
        self.assertEqual(response.status_code, 422)

    def test_v2_analise_diaria_sucesso_retorna_classificacao(self):
        """Cenário: a análise diária V2 recebe dados válidos.
        Resultado esperado: HTTP 200 com Data, Consumo e classificacao."""
        payload = {"sensor_id": "sensor_1"}
        historico = build_daily_history(count=30)

        with patch("appSM.api.views.EstatisticaUseCase") as mock_uc_cls:
            mock_uc_cls.diario.return_value = {
                "Data": "30/01/2024",
                "Consumo": 28.0,
                "Classificação": 1,
            }

            response = self.client.post(reverse("v2-classificacao-consumo-diaria"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Data": "30/01/2024", "Consumo": 28.0, "classificacao": 1})
        mock_uc_cls.diario.assert_called_once_with(sensor_id="sensor_1")

    def test_v2_analise_mensal_sucesso_retorna_classificacao(self):
        """Cenário: a análise mensal V2 recebe dados válidos.
        Resultado esperado: HTTP 200 com o mesmo contrato de saída da análise diária."""
        payload = {"unidade_id": 10}
        historico = build_monthly_history()

        with patch("appSM.api.views.EstatisticaUseCase") as mock_uc_cls:
            mock_uc_cls.mensal.return_value = {
                "Data": "01/12/2024",
                "Consumo": 111.0,
                "Classificação": 3,
            }

            response = self.client.post(reverse("v2-classificacao-consumo-mensal"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Data": "01/12/2024", "Consumo": 111.0, "classificacao": 3})
        mock_uc_cls.mensal.assert_called_once_with(unidade_id=10, dispositivo_id=None)

    def test_v2_dados_bandas_sucesso_retorna_lista_processada(self):
        """Cenário: o endpoint de bandas V2 recebe um sensor válido.
        Resultado esperado: HTTP 200 com a chave dados contendo uma lista de registros."""
        payload = {"sensor_id": "sensor_1"}
        historico = build_daily_history(count=6)

        with patch("appSM.api.views.EstatisticaUseCase") as mock_uc_cls:
            mock_uc_cls.dados_completos.return_value = [
                {"Data": "01/01/2024", "Consumo": 10.0, "Média Móvel": 10.0, "Desvio Padrão": 0.0}
            ]

            response = self.client.post(reverse("v2-dados-bandas"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("dados", body)
        self.assertIsInstance(body["dados"], list)
        self.assertEqual(body["dados"][0]["Data"], "01/01/2024")
        mock_uc_cls.dados_completos.assert_called_once_with(sensor_id="sensor_1")

    def test_classificacao_ph_sucesso_retorna_payload_do_servico(self):
        """Cenário: o endpoint de pH recebe client_id e ph_value válidos.
        Resultado esperado: HTTP 200 com o payload completo devolvido pelo serviço."""
        payload = {"client_id": "sisar", "ph_value": 7.2}

        with patch("appSM.api.views.PHClassificationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.classify.return_value = {
                "client_id": "sisar",
                "ph_value": 7.2,
                "classification": "adequado",
                "confidence": 0.95,
                "model_version": "v1.0.0",
            }

            response = self.client.post(reverse("classificacao-ph"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["classification"], "adequado")
        self.assertEqual(body["client_id"], "sisar")
        self.assertIsInstance(body["ph_value"], float)
        mock_service_cls.assert_called_once()
        mock_service.classify.assert_called_once_with(client_id="sisar", ph_value=7.2)

    def test_classificacao_ph_sem_client_id_retorna_400(self):
        """Cenário: o campo obrigatório client_id está ausente.
        Resultado esperado: HTTP 400 com mensagem explícita de campo faltante."""
        response = self.client.post(reverse("classificacao-ph"), {"ph_value": 7.2}, format="json")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Campo obrigatório ausente: client_id"})

    def test_classificacao_ph_sem_ph_value_retorna_400(self):
        """Cenário: o campo obrigatório ph_value está ausente.
        Resultado esperado: HTTP 400 com mensagem explícita de campo faltante."""
        response = self.client.post(reverse("classificacao-ph"), {"client_id": "sisar"}, format="json")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Campo obrigatório ausente: ph_value"})

    def test_classificacao_ph_tipo_invalido_retorna_422(self):
        """Cenário: ph_value não pode ser convertido para número.
        Resultado esperado: HTTP 422 com mensagem informando o tipo recebido."""
        response = self.client.post(
            reverse("classificacao-ph"),
            {"client_id": "sisar", "ph_value": "alto"},
            format="json",
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"error": "ph_value deve ser um número, recebido: alto"})

    def test_classificacao_ph_modelo_nao_encontrado_retorna_404(self):
        """Cenário: o serviço informa que o modelo do cliente não existe.
        Resultado esperado: HTTP 404 com erro e detalhe do problema."""
        payload = {"client_id": "sisar", "ph_value": 7.2}

        with patch("appSM.api.views.PHClassificationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.classify.side_effect = FileNotFoundError("arquivo ausente")

            response = self.client.post(reverse("classificacao-ph"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(body["error"], "Modelo não encontrado para este cliente")
        self.assertIn("arquivo ausente", body["detail"])


class ClassificationRangeAPITests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="api_tester2",
            password="strong-password-123",
        )
        self.client.force_authenticate(user=self.user)

    def test_v2_classification_range_returns_true(self):
        execution_id = "d7d746c8-c95f-4cb0-b004-dc4995f5ef56"
        payload = {
            "unidade_id": 10,
            "reference_period": "2026-07-27",
            "execution_id": execution_id,
        }
        service_result = {
            "outside_green_range": True,
            "severity": "critical",
            "classification": 2,
            "classification_label": "Consumo Excessivo",
            "reference_period": "2026-07-27",
            "execution_id": execution_id,
        }
        with patch("appSM.api.views.RangeUseCase") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = service_result
            
            response = self.client.post(reverse("v2-classification-range"), payload, format="json")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), service_result)
            mock_service.processar.assert_called_once_with(
                10,
                date(2026, 7, 27),
                UUID(execution_id),
            )

    def test_v2_classification_range_returns_false(self):
        payload = {"unidade_id": 10}
        service_result = {
            "outside_green_range": False,
            "severity": "green",
            "classification": 0,
            "classification_label": "Consumo Moderado",
            "reference_period": "2026-07-27",
        }
        with patch("appSM.api.views.RangeUseCase") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = service_result
            
            response = self.client.post(reverse("v2-classification-range"), payload, format="json")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), service_result)
            mock_service.processar.assert_called_once_with(10, None, None)

    def test_v2_classification_range_no_data(self):
        payload = {"unidade_id": 10}
        with patch("appSM.api.views.RangeUseCase") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            from appSM.application.exceptions import ConsumoNaoEncontrado
            mock_service.processar.side_effect = ConsumoNaoEncontrado("Nenhum registro encontrado no periodo solicitado")
            
            response = self.client.post(reverse("v2-classification-range"), payload, format="json")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"error": "Nenhum registro encontrado no periodo solicitado"})

    def test_v2_classification_range_rejects_invalid_reference_period(self):
        response = self.client.post(
            reverse("v2-classification-range"),
            {"unidade_id": 10, "reference_period": "27/07/2026"},
            format="json",
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("reference_period", response.json()["details"])


class RangeUseCaseTests(SimpleTestCase):
    def _test_classification_value(self, classification_value, expected_outside, expected_severity):
        with patch("appSM.application.range_use_case.HistoricoUseCase") as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {
                "results": [{"periodo": "27/07/2026", "consumo": 12.0, "classificacao": classification_value}]
            }
            
            service = RangeUseCase(historico_use_case=mock_history)
            resultado = service.processar(10, date(2026, 7, 27))
            
            self.assertEqual(resultado["outside_green_range"], expected_outside)
            self.assertEqual(resultado["severity"], expected_severity)
            self.assertEqual(resultado["classification"], classification_value)
            self.assertEqual(resultado["reference_period"], "2026-07-27")
            mock_history.processar.assert_called_once()
            args = mock_history.processar.call_args[0][0]
            self.assertEqual(args["type"], "daily")
            self.assertEqual(args["unidade_id"], 10)
            self.assertEqual(args["data_inicio"], date(2026, 7, 27))
            self.assertEqual(args["data_fim"], date(2026, 7, 27))

    def test_classification_minus_2(self):
        self._test_classification_value(-2, True, "critical")

    def test_classification_minus_1(self):
        self._test_classification_value(-1, False, "green")

    def test_classification_0(self):
        self._test_classification_value(0, False, "green")

    def test_classification_1(self):
        self._test_classification_value(1, True, "warning")

    def test_classification_2(self):
        self._test_classification_value(2, True, "critical")

    def test_no_data_raises_error(self):
        with patch("appSM.application.range_use_case.HistoricoUseCase") as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {"results": []}
            
            service = RangeUseCase(historico_use_case=mock_history)
            from appSM.infrastructure.exceptions import DataNotFoundError
            from appSM.application.exceptions import ConsumoNaoEncontrado
            with self.assertRaises(ConsumoNaoEncontrado):
                service.processar(10)

    def test_invalid_classification_raises_error(self):
        with patch("appSM.application.range_use_case.HistoricoUseCase") as mock_history_cls:
            mock_history = mock_history_cls.return_value
            mock_history.processar.return_value = {
                "results": [{"periodo": "27/07/2026", "consumo": 12.0, "classificacao": 3}]
            }

            with self.assertRaisesRegex(ValueError, "fora do intervalo esperado"):
                RangeUseCase(historico_use_case=mock_history).processar(10, date(2026, 7, 27))


from .test_characterization import *
