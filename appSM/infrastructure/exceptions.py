class DataNotFoundError(LookupError):
    """Dado não encontrado no repositório externo. Não vaza para fora da camada de application."""

class DeviceNotFoundError(LookupError):
    """Dispositivo não encontrado. Não vaza para fora da camada de application."""
