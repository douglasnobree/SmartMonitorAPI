"""
Testes para o serviço unificado de análise estatística.

Execute com: python -m pytest ml_pipeline/senseFlow_A/classificacao/test_analise_estatistica.py -v
"""

import pytest
from ml_pipeline.senseFlow_A.classificacao.analise_estatistica_service import (
    AnaliseEstatisticaService,
    analiseEstatisticaDiaria_service,
    analiseEstatisticaMensal_service,
)


class TestAnaliseEstatisticaService:
    """Testes para o serviço unificado."""
    
    def setup_method(self):
        """Setup executado antes de cada teste."""
        # Dados de exemplo
        self.dados_diarios = {
            f"{i:02d}/01/2025": 100 + i * 2 for i in range(1, 31)
        }
        
        self.dados_mensais = {
            f"01/{i:02d}/2024": 1000 + i * 50 for i in range(1, 13)
        }
    
    def test_servico_diario_janela_padrao(self):
        """Testa serviço com janela padrão (30 dias)."""
        service = AnaliseEstatisticaService()
        assert service.janela == 30
        
        resultado = service.processarDados(self.dados_diarios)
        
        assert 'Data' in resultado
        assert 'Consumo' in resultado
        assert 'Classificação' in resultado
    
    def test_servico_mensal_janela_customizada(self):
        """Testa serviço com janela customizada (12 meses)."""
        service = AnaliseEstatisticaService(janela=12)
        assert service.janela == 12
        
        resultado = service.processarDados(self.dados_mensais)
        
        assert resultado is not None
        assert isinstance(resultado, dict)
    
    def test_alias_diario_compatibilidade(self):
        """Testa alias para análise diária (compatibilidade)."""
        service = analiseEstatisticaDiaria_service()
        assert service.janela == 30
        
        resultado = service.processarDados(self.dados_diarios)
        assert resultado is not None
    
    def test_alias_mensal_compatibilidade(self):
        """Testa alias para análise mensal (compatibilidade)."""
        service = analiseEstatisticaMensal_service()
        assert service.janela == 12
        
        resultado = service.processarDados(self.dados_mensais)
        assert resultado is not None
    
    def test_janela_customizada_semanal(self):
        """Testa janela customizada para análise semanal."""
        service = AnaliseEstatisticaService(janela=7)
        assert service.janela == 7
        
        dados_semanais = {
            f"{i:02d}/01/2025": 50 + i for i in range(1, 15)
        }
        
        resultado = service.processarDados(dados_semanais)
        assert resultado is not None
    
    def test_dados_vazios(self):
        """Testa comportamento com dados vazios."""
        service = AnaliseEstatisticaService()
        
        with pytest.raises(ValueError, match="não pode estar vazio"):
            service.processarDados({})
    
    def test_classificacao_valores(self):
        """Testa se a classificação retorna valores válidos."""
        service = AnaliseEstatisticaService(janela=5)
        
        # Dados com variação para gerar classificações diferentes
        dados = {
            f"{i:02d}/01/2025": valor 
            for i, valor in enumerate([100, 102, 98, 101, 150], start=1)
        }
        
        resultado = service.processarDados(dados)
        classificacao = resultado['Classificação']
        
        # Deve ser um dos valores válidos ou "Sem classificação"
        valores_validos = [-2, -1, 0, 1, 2, "Sem classificação"]
        assert classificacao in valores_validos
    
    def test_consistencia_entre_servicos(self):
        """Testa se alias e serviço direto retornam mesmo resultado."""
        # Usando alias
        service_alias = analiseEstatisticaDiaria_service()
        resultado_alias = service_alias.processarDados(self.dados_diarios)
        
        # Usando serviço direto
        service_direto = AnaliseEstatisticaService(janela=30)
        resultado_direto = service_direto.processarDados(self.dados_diarios)
        
        # Devem retornar o mesmo resultado
        assert resultado_alias['Data'] == resultado_direto['Data']
        assert resultado_alias['Consumo'] == resultado_direto['Consumo']
        assert resultado_alias['Classificação'] == resultado_direto['Classificação']


class TestConstantes:
    """Testes para constantes do serviço."""
    
    def test_constantes_janela(self):
        """Testa se as constantes estão definidas corretamente."""
        assert AnaliseEstatisticaService.JANELA_DIARIA == 30
        assert AnaliseEstatisticaService.JANELA_MENSAL == 12
    
    def test_alias_usam_constantes(self):
        """Testa se aliases usam as constantes corretas."""
        service_diario = analiseEstatisticaDiaria_service()
        service_mensal = analiseEstatisticaMensal_service()
        
        assert service_diario.janela == AnaliseEstatisticaService.JANELA_DIARIA
        assert service_mensal.janela == AnaliseEstatisticaService.JANELA_MENSAL


if __name__ == "__main__":
    # Permite executar o teste diretamente
    pytest.main([__file__, "-v"])
