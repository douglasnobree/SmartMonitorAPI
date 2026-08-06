class ConsumoNaoEncontrado(Exception):
    """Não há histórico de consumo para os filtros informados."""
    http_status = 404

class DispositivoNaoEncontrado(Exception):
    """Dispositivo referenciado não existe no sistema."""
    http_status = 404
