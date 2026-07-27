from datetime import datetime
from rest_framework import serializers

class MySerializer(serializers.Serializer):
    def to_internal_value(self, data):
        """ Validação legado... """
        if not isinstance(data, dict):
            raise serializers.ValidationError("Os dados devem ser um dicionário.")
        for key, value in data.items():
            try:
                datetime.strptime(key, '%d/%m/%Y')
            except ValueError:
                raise serializers.ValidationError(f"A chave '{key}' não está no formato DD/MM/YYYY.")
            if value is not None and not isinstance(value, (float, int)):
                raise serializers.ValidationError(f"O valor para a chave '{key}' deve ser um número ou null.")
        return super().to_internal_value(data)
        
class V2DailySerializer(serializers.Serializer):
    """Payload para análises diárias baseadas no sensor."""
    sensor_id = serializers.CharField(max_length=11)

class V2MonthlySerializer(serializers.Serializer):
    """Payload para análises mensais baseadas na unidade e no dispositivo (ciclo automático)."""
    unidade_id = serializers.IntegerField(min_value=1)
    # Substituído: agora recebemos o ID do dispositivo para buscar o fechamento no banco
    dispositivo_id = serializers.CharField(max_length=50, required=False)

    def validate(self, attrs):
        # Validações básicas adicionais podem ser inseridas aqui se necessário
        return attrs


class V2ClassificationHistorySerializer(serializers.Serializer):
    """Payload para o relatorio historico de classificacao."""

    type = serializers.ChoiceField(choices=("daily", "monthly"))
    unidade_id = serializers.IntegerField(min_value=1)
    data_inicio = serializers.DateField(required=False)
    data_fim = serializers.DateField(required=False)
    ano = serializers.IntegerField(min_value=1900, max_value=3000, required=False)
    dispositivo_id = serializers.CharField(max_length=50, required=False)

    def validate(self, attrs):
        tipo = attrs.get("type")

        if tipo == "daily":
            if "data_inicio" not in attrs or "data_fim" not in attrs:
                raise serializers.ValidationError(
                    "data_inicio e data_fim sao obrigatorios para type=daily."
                )
            if attrs["data_inicio"] > attrs["data_fim"]:
                raise serializers.ValidationError("data_inicio deve ser menor ou igual a data_fim.")

        if tipo == "monthly" and "ano" not in attrs:
            raise serializers.ValidationError("ano e obrigatorio para type=monthly.")

        return attrs
