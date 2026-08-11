from datetime import date

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
