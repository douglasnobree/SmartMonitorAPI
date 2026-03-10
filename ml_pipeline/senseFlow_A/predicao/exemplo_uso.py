"""
Exemplos de uso do PredicaoService com Inversão de Dependência.

Este arquivo demonstra como usar o serviço de predição com:
1. Modelo padrão (sem injeção de dependência)
2. Modelo customizado (com injeção de dependência)
3. Mock para testes unitários
"""

from predicao_service import PredicaoService
from ..modelos.regressaoLinear import LinearRegression_Acumulado
from ml_pipeline.modelos.base_modelo import ModeloPredicao
import pandas as pd


# ========== Exemplo 1: Uso Padrão (sem injeção) ==========
def exemplo_uso_padrao():
    """Uso padrão do serviço com modelo padrão LinearRegression_Acumulado."""
    
    dados_historicos = {
        '01/01/2024': 100.5,
        '02/01/2024': 120.3,
        '03/01/2024': 115.8,
        '04/01/2024': 130.2,
        '05/01/2024': 125.7
    }
    
    # Serviço usa modelo padrão automaticamente
    service = PredicaoService(tipo='diaria')
    resultado = service.processarDados(dados_historicos)
    
    print(f"Predição diária (modelo padrão): {resultado:.2f}")


# ========== Exemplo 2: Injeção de Dependência ==========
def exemplo_com_injecao():
    """Uso do serviço com modelo customizado injetado."""
    
    dados_historicos = {
        '01/01/2024': 1000.0,
        '01/02/2024': 1200.0,
        '01/03/2024': 1150.0,
        '01/04/2024': 1300.0,
        '01/05/2024': 1250.0
    }
    
    # Criar modelo customizado com tipo_predicao configurado
    modelo_custom = LinearRegression_Acumulado(tipo_predicao='mensal')
    
    # Injetar modelo no serviço
    service = PredicaoService(tipo='mensal', modelo=modelo_custom)
    resultado = service.processarDados(dados_historicos)
    
    print(f"Predição mensal (modelo injetado): {resultado:.2f}")


# ========== Exemplo 3: Mock para Testes ==========
def exemplo_com_mock():
    """Exemplo de teste unitário usando mock do modelo."""
    from unittest.mock import Mock
    
    # Criar mock do modelo
    modelo_mock = Mock(spec=ModeloPredicao)
    modelo_mock.prever.return_value = 150.0
    
    # Injetar mock no serviço
    service = PredicaoService(tipo='diaria', modelo=modelo_mock)
    
    dados_teste = {'01/01/2024': 100.0, '02/01/2024': 110.0}
    resultado = service.processarDados(dados_teste)
    
    # Verificar que modelo foi chamado corretamente
    modelo_mock.treinar.assert_called_once()
    modelo_mock.prever.assert_called_once_with(2)
    
    print(f"Predição com mock: {resultado}")
    print("✅ Mock funcionou corretamente!")


# ========== Exemplo 4: Criar Modelo Customizado ==========
class ModeloConstante(ModeloPredicao):
    """Modelo simples que sempre retorna um valor constante (para testes)."""
    
    def __init__(self, valor_constante=100.0, tipo_predicao=None):
        self.valor = valor_constante
        self.tipo_predicao = tipo_predicao
    
    def treinar(self, dados_brutos: pd.DataFrame) -> None:
        """Este modelo não precisa treinar com dados."""
        print(f"Modelo constante 'treinado' com {len(dados_brutos)} registros")
    
    def prever(self, n_passos: int) -> float:
        """Retorna valor constante com ajuste baseado no tipo."""
        base = self.valor
        
        # Aplicar ajuste baseado no tipo_predicao
        if self.tipo_predicao == 'mensal':
            base *= 1.1  # 10% maior para mensal
        elif self.tipo_predicao == 'diaria':
            base *= 0.9  # 10% menor para diária
        
        return base


def exemplo_modelo_customizado():
    """Exemplo usando modelo completamente customizado."""
    
    # Modelo simples sem tipo
    modelo_simples = ModeloConstante(valor_constante=200.0)
    service = PredicaoService(tipo='diaria', modelo=modelo_simples)
    
    dados = {'01/01/2024': 100.0, '02/01/2024': 150.0}
    resultado = service.processarDados(dados)
    
    print(f"Predição com modelo customizado: {resultado}")


if __name__ == '__main__':
    print("=" * 60)
    print("EXEMPLOS DE USO - Inversão de Dependência")
    print("=" * 60)
    
    print("\n1. Uso Padrão:")
    exemplo_uso_padrao()
    
    print("\n2. Com Injeção de Dependência:")
    exemplo_com_injecao()
    
    print("\n3. Com Mock (Testes):")
    exemplo_com_mock()
    
    print("\n4. Modelo Customizado:")
    exemplo_modelo_customizado()
    
    print("\n" + "=" * 60)
    print("✅ Todos os exemplos executados com sucesso!")
    print("=" * 60)
