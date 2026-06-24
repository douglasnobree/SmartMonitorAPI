from datetime import date
from unittest.mock import patch

import pandas as pd

from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import SimpleTestCase, TestCase

from rest_framework import serializers
from rest_framework.test import APIClient, APITestCase

from appSM.serializers import MySerializer
from appSM.services.classification_history_service import ClassificationHistoryService
from ml_pipeline.senseFlow_A.predicao.predicao_service import PredicaoService
from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import (
    AnaliseEstatisticaService,
)


def build_daily_history(count=5, start_year=2024, start_month=1, start_day=1, base_value=10.0):
    payload = {}
    current_date = date(start_year, start_month, start_day)

    for index in range(count):
        payload[current_date.strftime("%d/%m/%Y")] = float(base_value + index)
        current_date = date.fromordinal(current_date.toordinal() + 1)
    
    print("Payload de teste diário gerado:", payload)  # Debug: exibe o payload gerado
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


class PredicaoServiceTests(SimpleTestCase):
    def test_processar_dados_validos_treina_modelo_e_retorna_float(self):
        """Cenário: histórico válido é enviado ao serviço.
        Resultado esperado: o serviço normaliza os dados, treina o modelo mockado e retorna um float."""
        with patch("ml_pipeline.senseFlow_A.predicao.predicao_service.LinearRegressionAcumulado") as mock_model_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.return_value = 12.75

            service = PredicaoService(tipo="diaria")
            resultado = service.processarDados(build_daily_history())

            self.assertIsInstance(resultado, float)
            self.assertEqual(resultado, 12.75)
            mock_model.treinar.assert_called_once()
            mock_model.prever.assert_called_once()

            dataframe_enviado = mock_model.treinar.call_args.args[0]
            self.assertEqual(list(dataframe_enviado.columns), ["Data", "Consumo"])
            self.assertEqual(len(dataframe_enviado), 5)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: o payload chega vazio.
        Resultado esperado: o serviço interrompe o fluxo com ValueError sem chamar o modelo."""
        service = PredicaoService(tipo="diaria")

        with self.assertRaisesMessage(ValueError, "dados_request não pode estar vazio"):
            service.processarDados({})

    def test_processar_dados_com_datas_invalidas_gera_value_error(self):
        """Cenário: todas as datas recebidas são inválidas.
        Resultado esperado: a normalização falha com ValueError antes de treinar qualquer modelo."""
        with patch("ml_pipeline.senseFlow_A.predicao.predicao_service.LinearRegressionAcumulado") as mock_model_cls:
            service = PredicaoService(tipo="diaria")

            with self.assertRaisesMessage(ValueError, "Nenhuma data valida encontrada no historico"):
                service.processarDados({"31/02/2024": 10.0})

            mock_model_cls.return_value.treinar.assert_not_called()

    def test_processar_dados_propaga_excecao_inesperada_do_modelo(self):
        """Cenário: o modelo mockado quebra durante a predição.
        Resultado esperado: o serviço converte a falha em Exception para a camada superior."""
        with patch("ml_pipeline.senseFlow_A.predicao.predicao_service.LinearRegressionAcumulado") as mock_model_cls:
            mock_model = mock_model_cls.return_value
            mock_model.prever.side_effect = RuntimeError("falha na inferencia")

            service = PredicaoService(tipo="diaria")

            with self.assertRaisesMessage(Exception, "falha na inferencia"):
                service.processarDados(build_daily_history())

    def test_tratar_outliers_mediana_nao_altera_variacao_leve(self):
        """Cenário: a série tem apenas uma oscilação pequena.
        Resultado esperado: nenhum ponto é marcado como outlier."""
        service = PredicaoService(tipo="diaria")
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

        tratado, mascara = service._tratar_outliers_mediana(df.copy())

        self.assertEqual(int(mascara.sum()), 0)
        self.assertTrue(tratado["Consumo"].equals(df["Consumo"]))

    def test_tratar_outliers_mediana_trata_pico_extremo(self):
        """Cenário: a série contém um pico muito acima do padrão.
        Resultado esperado: apenas o pico é substituído pela mediana."""
        service = PredicaoService(tipo="diaria")
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

        tratado, mascara = service._tratar_outliers_mediana(df.copy())

        self.assertEqual(int(mascara.sum()), 1)
        self.assertEqual(float(tratado.loc[mascara, "Consumo"].iloc[0]), 101.5)


class AnaliseEstatisticaServiceTests(SimpleTestCase):
    def test_processar_dados_validos_retorna_dict_com_estrutura_esperada(self):
        """Cenário: um histórico mensal válido é enviado à análise estatística.
        Resultado esperado: o serviço retorna um dicionário com Data, Consumo e Classificação."""
        service = AnaliseEstatisticaService(janela=12)
        resultado = service.processarDados(build_monthly_history())

        self.assertIsInstance(resultado, dict)
        self.assertEqual(set(resultado.keys()), {"Data", "Consumo", "Classificação"})
        self.assertIsInstance(resultado["Data"], str)
        self.assertIsInstance(resultado["Consumo"], float)
        self.assertIsInstance(resultado["Classificação"], str)

    def test_obter_dados_completos_retornam_lista_de_dicionarios(self):
        """Cenário: o serviço recebe um histórico válido para bandas completas.
        Resultado esperado: a resposta é uma lista de registros com colunas processadas e tipos consistentes."""
        service = AnaliseEstatisticaService(janela=30)
        resultado = service.obterDadosCompletos(build_daily_history(count=8))

        self.assertIsInstance(resultado, list)
        self.assertGreater(len(resultado), 0)
        primeiro = resultado[0]
        self.assertIsInstance(primeiro, dict)
        self.assertIn("Data", primeiro)
        self.assertIn("Consumo", primeiro)
        self.assertIn("Média Móvel", primeiro)
        self.assertIn("Desvio Padrão", primeiro)

    def test_processar_dados_vazio_gera_value_error(self):
        """Cenário: a análise recebe um payload vazio.
        Resultado esperado: o serviço rejeita a requisição com ValueError."""
        service = AnaliseEstatisticaService(janela=30)

        with self.assertRaisesMessage(ValueError, "dados_request não pode estar vazio"):
            service.processarDados({})


class ClassificationHistoryServiceTests(SimpleTestCase):
    def test_daily_usa_contexto_e_retorna_apenas_periodo_solicitado(self):
        """Cenario: ha dados anteriores ao periodo solicitado.
        Resultado esperado: contexto entra no pipeline, mas a resposta fica restrita a janela."""

        class FakeFetcher:
            def fetch_history_daily_report(self, unidade_id, data_inicio, data_fim):
                index = pd.date_range("2026-05-30", "2026-06-03", freq="D")
                return pd.DataFrame({"Consumo": [7.0, 8.0, 9.0, 10.0, 11.0]}, index=index)

        class FakeAnalysisService:
            calls = []

            def __init__(self, janela):
                self.janela = janela

            def processarDados(self, historico):
                self.calls.append(historico)
                ultima_data, ultimo_consumo = list(historico.items())[-1]
                return {"Data": ultima_data, "Consumo": ultimo_consumo, "Classificação": -2}

        service = ClassificationHistoryService(
            fetcher=FakeFetcher(),
            analysis_service_cls=FakeAnalysisService,
        )

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
        self.assertEqual(resultado["results"][0]["classificacao"], "Economia Máxima")
        self.assertIn("30/05/2026", FakeAnalysisService.calls[0])
        self.assertNotIn("31/05/2026", [item["periodo"] for item in resultado["results"]])

    def test_periodos_mensais_respeitam_dia_fechamento(self):
        """Cenario: o fechamento configurado e dia 14.
        Resultado esperado: os ciclos do ano terminam no dia anterior do mes de referencia."""
        periodos = ClassificationHistoryService._periodos_do_ano(2026, 14)

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

    def test_predicao_diaria_sucesso_retornando_prediction(self):
        """Cenário: o endpoint diário recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = build_daily_history()

        with patch("appSM.views.PredicaoService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processarDados.return_value = 19.5

            response = self.client.post(reverse("predicao-consumo-diario"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Prediction": 19.5})
        mock_service_cls.assert_called_once_with(tipo="diaria")
        mock_service.processarDados.assert_called_once_with(payload)

    def test_predicao_mensal_sucesso_retornando_prediction(self):
        """Cenário: o endpoint mensal recebe um JSON válido.
        Resultado esperado: HTTP 200 com a chave Prediction e valor numérico."""
        payload = build_monthly_history()

        with patch("appSM.views.PredicaoService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processarDados.return_value = 220.0

            response = self.client.post(reverse("predicao-consumo-mensal"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Prediction": 220.0})
        mock_service_cls.assert_called_once_with(tipo="mensal")
        mock_service.processarDados.assert_called_once_with(payload)

    def test_predicao_diaria_payload_vazio_retorna_400(self):
        """Cenário: a requisição chega sem body.
        Resultado esperado: HTTP 400 com mensagem de body vazio."""
        response = self.client.generic(
            "POST",
            reverse("predicao-consumo-diario"),
            data="",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "Body da requisição está vazio"})

    def test_predicao_mensal_json_malformado_retorna_400(self):
        """Cenário: o JSON enviado é inválido.
        Resultado esperado: HTTP 400 com mensagem de JSON mal formatado."""
        response = self.client.generic(
            "POST",
            reverse("predicao-consumo-mensal"),
            data="{invalid-json",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "JSON mal formatado. Verifique a sintaxe."})

    def test_predicao_diaria_payload_nao_dict_retorna_422(self):
        """Cenário: o payload é um array JSON em vez de um objeto.
        Resultado esperado: HTTP 422 com mensagem de estrutura inválida."""
        response = self.client.post(reverse("predicao-consumo-diario"), [1, 2, 3], format="json")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"error": "Dados devem ser um objeto JSON não vazio com datas e valores"},
        )

    def test_predicao_diaria_erro_interno_retorna_500(self):
        """Cenário: o serviço lança uma exceção inesperada.
        Resultado esperado: HTTP 500 com mensagem genérica de erro interno."""
        payload = build_daily_history()

        with patch("appSM.views.PredicaoService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processarDados.side_effect = Exception("falha inesperada")

            response = self.client.post(reverse("predicao-consumo-diario"), payload, format="json")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"error": "Erro interno ao processar predição. Tente novamente."},
        )

    def test_predicao_diaria_exige_autenticacao(self):
        """Cenário: a rota é chamada sem autenticação.
        Resultado esperado: HTTP 401 antes de qualquer execução da view."""
        anon_client = APIClient()
        response = anon_client.post(reverse("predicao-consumo-diario"), build_daily_history(), format="json")

        self.assertEqual(response.status_code, 401)

    def test_v2_classification_history_daily_sucesso(self):
        """Cenario: relatorio historico diario recebe filtros validos.
        Resultado esperado: HTTP 200 com results retornado pelo servico."""
        payload = {
            "type": "daily",
            "unidade_id": 10,
            "data_inicio": "2026-06-01",
            "data_fim": "2026-06-03",
        }

        with patch("appSM.v2_views.ClassificationHistoryService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processar.return_value = {
                "results": [
                    {"periodo": "01/06/2026", "consumo": 9.34, "classificacao": "Economia Máxima"}
                ]
            }

            response = self.client.post(reverse("v2-classification-history"), payload, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["periodo"], "01/06/2026")
        mock_service.processar.assert_called_once()
        self.assertEqual(mock_service.processar.call_args.args[0]["unidade_id"], 10)

    def test_v2_classification_history_periodo_invalido_retorna_422(self):
        """Cenario: data_inicio e maior que data_fim.
        Resultado esperado: HTTP 422 seguindo o padrao v2 de payload invalido."""
        payload = {
            "type": "daily",
            "unidade_id": 10,
            "data_inicio": "2026-06-30",
            "data_fim": "2026-06-01",
        }

        response = self.client.post(reverse("v2-classification-history"), payload, format="json")

        self.assertEqual(response.status_code, 422)

    def test_analise_diaria_sucesso_retorna_classificacao(self):
        """Cenário: a análise diária recebe dados válidos.
        Resultado esperado: HTTP 200 com Data, Consumo e classificacao."""
        payload = build_daily_history(count=30)

        with patch("appSM.views.AnaliseEstatisticaService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processarDados.return_value = {
                "Data": "30/01/2024",
                "Consumo": 28.0,
                "Classificação": 1,
            }

            response = self.client.post(reverse("classificacao-consumo-diaria"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Data": "30/01/2024", "Consumo": 28.0, "classificacao": 1})
        mock_service_cls.assert_called_once_with(janela=30)
        mock_service.processarDados.assert_called_once_with(payload)

    def test_analise_mensal_sucesso_retorna_classificacao(self):
        """Cenário: a análise mensal recebe dados válidos.
        Resultado esperado: HTTP 200 com o mesmo contrato de saída da análise diária."""
        payload = build_monthly_history()

        with patch("appSM.views.AnaliseEstatisticaService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.processarDados.return_value = {
                "Data": "01/12/2024",
                "Consumo": 111.0,
                "Classificação": 3,
            }

            response = self.client.post(reverse("classificacao-consumo-mensal"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"Data": "01/12/2024", "Consumo": 111.0, "classificacao": 3})
        mock_service_cls.assert_called_once_with(janela=12)
        mock_service.processarDados.assert_called_once_with(payload)

    def test_dados_bandas_sucesso_retorna_lista_processada(self):
        """Cenário: o endpoint de bandas recebe um histórico válido.
        Resultado esperado: HTTP 200 com a chave dados contendo uma lista de registros."""
        payload = build_daily_history(count=6)

        with patch("appSM.views.AnaliseEstatisticaService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.obterDadosCompletos.return_value = [
                {"Data": "01/01/2024", "Consumo": 10.0, "Média Móvel": 10.0, "Desvio Padrão": 0.0}
            ]

            response = self.client.post(reverse("dados-bandas"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("dados", body)
        self.assertIsInstance(body["dados"], list)
        self.assertEqual(body["dados"][0]["Data"], "01/01/2024")
        mock_service_cls.assert_called_once_with(janela=30)
        mock_service.obterDadosCompletos.assert_called_once_with(payload)

    def test_classificacao_ph_sucesso_retorna_payload_do_servico(self):
        """Cenário: o endpoint de pH recebe client_id e ph_value válidos.
        Resultado esperado: HTTP 200 com o payload completo devolvido pelo serviço."""
        payload = {"client_id": "sisar", "ph_value": 7.2}

        with patch("appSM.views.PHClassificationService") as mock_service_cls:
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

        with patch("appSM.views.PHClassificationService") as mock_service_cls:
            mock_service = mock_service_cls.return_value
            mock_service.classify.side_effect = FileNotFoundError("arquivo ausente")

            response = self.client.post(reverse("classificacao-ph"), payload, format="json")

        body = response.json()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(body["error"], "Modelo não encontrado para este cliente")
        self.assertIn("arquivo ausente", body["detail"])


