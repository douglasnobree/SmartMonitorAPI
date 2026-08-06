from appSM.infrastructure.consumo_repository import ConsumoRepository
from appSM.infrastructure.dispositivo_repository import DispositivoRepository
from appSM.infrastructure.exceptions import DataNotFoundError as ExternalDataNotFoundError
from appSM.infrastructure.exceptions import DeviceNotFoundError as ExternalDeviceNotFoundError

class ExternalDataFetcher:
    """Proxy temporário para garantir que os testes continuem passando até a Fase 4."""
    def __init__(self, engine=None):
        self.consumo_repo = ConsumoRepository(engine)
        self.dispositivo_repo = DispositivoRepository(engine)
        
    def fetch_daily_history(self, sensor_id):
        return self.consumo_repo.buscar_historico_diario(sensor_id)
        
    def fetch_monthly_history(self, unidade_id, dispositivo_id=None):
        return self.consumo_repo.buscar_historico_mensal(unidade_id, dispositivo_id)
        
    def fetch_history_daily_report(self, unidade_id, data_inicio, data_fim):
        return self.consumo_repo.buscar_historico_relatorio_diario(unidade_id, data_inicio, data_fim)
        
    def fetch_history_monthly_report(self, unidade_id, data_inicio, data_fim):
        return self.consumo_repo.buscar_historico_mensal_bruto(unidade_id, data_inicio, data_fim)
        
    def fetch_dispositivo_dia_fechamento(self, dispositivo_id):
        return self.dispositivo_repo.buscar_dia_fechamento(dispositivo_id)
