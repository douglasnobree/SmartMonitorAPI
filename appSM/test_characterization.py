from datetime import date
from django.test import SimpleTestCase
import pandas as pd
import numpy as np

from appSM.domain import LinearRegressionAcumulado
from appSM.domain.outliers import tratar_outliers_iqr
from appSM.domain.bollinger import RollingWindowBollinger
from appSM.domain.classificador import classificar_por_faixa
from appSM.domain.ciclo_faturamento import agregar_por_ciclo_mensal, periodos_do_ano
from appSM.domain.tratamento import normalizar_historico


class CharacterizationTests(SimpleTestCase):
    """
    Testes de caracterização (Fase 2) para congelar o comportamento exato dos pipelines,
    cálculos estatísticos, normalização de dados e modelos de Machine Learning.
    """

    def test_char_normalizar_historico_preenche_gaps(self):
        """1. _normalizar_historico: parsing de datas, gaps e preenchimento com mediana."""
        payload = {"01/01/2024": 10.0, "03/01/2024": 30.0}
        df = normalizar_historico(payload, frequencia="diaria")
        
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["Data"]), ["01/01/2024", "02/01/2024", "03/01/2024"])
        self.assertEqual(float(df.loc[1, "Consumo"]), 20.0)

    def test_char_normalizar_historico_mensal(self):
        """2. _normalizar_historico com frequencia='mensal': DateOffset mensal."""
        payload = {"01/01/2024": 100.0, "01/03/2024": 300.0}
        df = normalizar_historico(payload, frequencia="mensal")
        
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["Data"]), ["01/01/2024", "01/02/2024", "01/03/2024"])
        self.assertEqual(float(df.loc[1, "Consumo"]), 200.0)

    def test_char_tratar_outliers_media_iqr_15(self):
        """3. _tratar_outliers_media: IQR 1.5×, substituição por média."""
        df = pd.DataFrame({
            "Data": ["01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024"],
            "Consumo": [10.0, 11.0, 12.0, 13.0, 100.0]
        })
        df_tratado, mascara = tratar_outliers_iqr(df.copy(), multiplicador=1.5, substituicao="media")
        
        self.assertTrue(mascara.iloc[4])
        self.assertAlmostEqual(float(df_tratado.loc[4, "Consumo"]), 11.5)

    def test_char_tratar_outliers_mediana_iqr_30(self):
        """4. _tratar_outliers_mediana: IQR 3.0×, substituição por mediana."""
        df_sem_outlier = pd.DataFrame({
            "Data": ["01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024"],
            "Consumo": [10.0, 11.0, 12.0, 13.0, 18.0]
        })
        _, mascara1 = tratar_outliers_iqr(df_sem_outlier.copy(), multiplicador=3.0, substituicao="mediana")
        self.assertFalse(mascara1.any())

        df_com_outlier = pd.DataFrame({
            "Data": ["01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024"],
            "Consumo": [10.0, 11.0, 12.0, 13.0, 100.0]
        })
        df_tratado, mascara2 = tratar_outliers_iqr(df_com_outlier.copy(), multiplicador=3.0, substituicao="mediana")
        self.assertTrue(mascara2.iloc[4])
        self.assertAlmostEqual(float(df_tratado.loc[4, "Consumo"]), 11.5)

    def test_char_calcular_bandas_bollinger(self):
        """5. _calcular_bandas: rolling mean/std, ±1σ, ±2σ, ±3σ e clip no 0."""
        estrategia = RollingWindowBollinger()
        df = pd.DataFrame({
            "Data": ["01/01/2024", "02/01/2024"],
            "Consumo": [10.0, 20.0]
        })
        df_bandas = estrategia.calcular(df, 2)
        
        std_val = np.std([10.0, 20.0], ddof=1)
        self.assertAlmostEqual(df_bandas.loc[1, "Média Móvel"], 15.0)
        self.assertAlmostEqual(df_bandas.loc[1, "Desvio Padrão"], std_val)
        self.assertAlmostEqual(df_bandas.loc[1, "Banda Sup 1"], 15.0 + 1 * std_val)
        self.assertAlmostEqual(df_bandas.loc[1, "Banda Sup 2"], 15.0 + 2 * std_val)
        self.assertAlmostEqual(df_bandas.loc[1, "Banda Sup 3"], 15.0 + 3 * std_val)
        self.assertAlmostEqual(df_bandas.loc[1, "Banda Inf 1"], 15.0 - 1 * std_val)

    def test_char_classificar_consumo_por_faixa(self):
        """6. Mapeamento valor→classe (-2,-1,0,1,2)."""
        classificar = classificar_por_faixa
        self.assertEqual(classificar(120.0, 120, 110, 90, 80), 2)
        self.assertEqual(classificar(115.0, 120, 110, 90, 80), 1)
        self.assertEqual(classificar(100.0, 120, 110, 90, 80), 0)
        self.assertEqual(classificar(85.0, 120, 110, 90, 80), -1)
        self.assertEqual(classificar(79.9, 120, 110, 90, 80), -2)

    def test_char_aggregate_monthly_ciclo_faturamento(self):
        """7. _aggregate_monthly: offset, resample MS, reconstituição."""
        dates = pd.date_range(start="2024-04-14", end="2024-05-13", freq="D")
        df = pd.DataFrame({"Consumo": [10.0] * len(dates)}, index=dates)
        
        mensal = agregar_por_ciclo_mensal(df, dia_inicio_ciclo=14)
        self.assertEqual(len(mensal), 1)
        self.assertEqual(mensal.index[0], pd.Timestamp("2024-04-14"))
        self.assertAlmostEqual(float(mensal.iloc[0]["Consumo"]), 300.0)

    def test_char_periodos_do_ano(self):
        """8. _periodos_do_ano: geração de 12 períodos com dia_fechamento."""
        periodos_14 = periodos_do_ano(2026, 14)
        self.assertEqual(len(periodos_14), 12)
        self.assertEqual(periodos_14[0], (date(2025, 12, 14), date(2026, 1, 13)))
        self.assertEqual(periodos_14[1], (date(2026, 1, 14), date(2026, 2, 13)))

        periodos_1 = periodos_do_ano(2026, 1)
        self.assertEqual(len(periodos_1), 12)
        self.assertEqual(periodos_1[0], (date(2026, 1, 1), date(2026, 1, 31)))

    def test_char_linear_regression_acumulado_diaria(self):
        """9. treinar/prever diária: cumsum, ajuste negativo, floor 0."""
        modelo = LinearRegressionAcumulado(tipo_predicao="diaria")
        df = pd.DataFrame({"Consumo": [10.0, 10.0, 10.0, 10.0]})
        modelo.treinar(df)
        
        previsao = modelo.prever(len(df))
        self.assertAlmostEqual(previsao, 10.0)
        self.assertGreaterEqual(previsao, 0.0)

    def test_char_linear_regression_acumulado_mensal(self):
        """10. treinar/prever mensal: ajuste sigma e abs."""
        modelo = LinearRegressionAcumulado(tipo_predicao="mensal")
        df = pd.DataFrame({"Consumo": [100.0, 105.0, 95.0, 100.0]})
        modelo.treinar(df)
        
        previsao = modelo.prever(len(df))
        self.assertIsInstance(previsao, float)
        self.assertGreaterEqual(previsao, 0.0)
