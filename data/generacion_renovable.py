import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

MADRID_TZ = ZoneInfo('Europe/Madrid')

# Indicadores ESIOS de "Generación medida" (real, no programada). Están
# desagregados por provincia, así que sumamos todas las provincias para
# obtener el total nacional en cada instante.
#
# A diferencia del indicador de precio (magnitud intensiva: €/MWh, hay que
# promediar), estos indicadores son de magnitud "Energía" (aditiva), así que
# sumar los 4 cuartos de hora para obtener el total horario SÍ es la
# operación correcta: no sufren el problema de precios_omie.py.
INDICATOR_EOLICA = 10037  # Generación medida Eólica (terrestre + marina)
INDICATOR_SOLAR = 10205   # Generación medida solar (fotovoltaica + térmica)


def _monthly_chunks(start_date_utc, end_date_utc):
    """Divide un rango [start, end] en tramos mensuales (start, end) en UTC.

    Estos indicadores están desagregados por ~45 provincias; pedir de golpe
    3 meses de datos hace que la API de ESIOS supere el timeout del gateway
    (504). Troceando por mes cada petición es rápida y fiable.
    """
    chunk_start = start_date_utc
    while chunk_start < end_date_utc:
        next_month = (chunk_start.replace(day=1) + pd.DateOffset(months=1))
        chunk_end = min(next_month, end_date_utc)
        yield chunk_start, chunk_end
        chunk_start = chunk_end


def _fetch_indicador_nacional(api_key, indicator_id, start_date_utc, end_date_utc, resolucion):
    """Descarga un indicador ESIOS desagregado por provincia y devuelve la
    serie nacional (suma de provincias) indexada por datetime en hora Madrid.
    """
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    headers = {
        'Accept': 'application/json; application/vnd.esios-api-v1+json',
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    series_chunks = []
    for chunk_start, chunk_end in _monthly_chunks(start_date_utc, end_date_utc):
        params = {
            'start_date': chunk_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end_date': chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time_trunc': resolucion
        }

        response = requests.get(url, headers=headers, params=params, timeout=90)
        if response.status_code != 200:
            print(f"Error en la petición al indicador {indicator_id} "
                  f"({chunk_start} a {chunk_end}): {response.status_code}")
            continue

        valores = response.json().get('indicator', {}).get('values', [])
        if not valores:
            continue

        df = pd.DataFrame(valores)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert(MADRID_TZ)
        series_chunks.append(df.groupby('datetime')['value'].sum())

    if not series_chunks:
        return pd.Series(dtype=float)

    return pd.concat(series_chunks).sort_index()


def obtener_generacion_renovable(api_key, start_date, end_date, resolucion='hour'):
    """
    Obtiene la generación eólica y solar medida (real) a nivel nacional
    desde ESIOS, sumando las provincias.

    Parámetros:
    - api_key (str): Tu token personal de la API de ESIOS.
    - start_date (datetime): Fecha y hora de inicio en hora local de Madrid
      (si no lleva tzinfo, se asume Europe/Madrid).
    - end_date (datetime): Fecha y hora de fin en hora local de Madrid
      (si no lleva tzinfo, se asume Europe/Madrid).
    - resolucion (str): 'hour' o 'minutes15'.

    Devuelve un DataFrame con columnas: date_aware, date, period,
    wind_mwh, solar_mwh, renewable_total_mwh.
    """
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=MADRID_TZ)
    else:
        start_date = start_date.astimezone(MADRID_TZ)

    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=MADRID_TZ)
    else:
        end_date = end_date.astimezone(MADRID_TZ)

    start_date_utc = start_date.astimezone(timezone.utc)
    end_date_utc = end_date.astimezone(timezone.utc)

    print(f"Consultando generación renovable desde {start_date} hasta {end_date} (hora Madrid)...")
    eolica = _fetch_indicador_nacional(api_key, INDICATOR_EOLICA, start_date_utc, end_date_utc, resolucion)
    solar = _fetch_indicador_nacional(api_key, INDICATOR_SOLAR, start_date_utc, end_date_utc, resolucion)

    if eolica.empty and solar.empty:
        print("La consulta fue exitosa, pero no hay datos para este periodo.")
        return pd.DataFrame()

    df = pd.DataFrame({'wind_mwh': eolica, 'solar_mwh': solar})
    df.index.name = 'datetime'
    df = df.reset_index().sort_values('datetime')

    # Ausencia de filas (p. ej. solar de madrugada) equivale a generación nula.
    df['wind_mwh'] = df['wind_mwh'].fillna(0.0)
    df['solar_mwh'] = df['solar_mwh'].fillna(0.0)
    df['renewable_total_mwh'] = df['wind_mwh'] + df['solar_mwh']

    df_final = pd.DataFrame({
        'date_aware': df['datetime'],
        'date': df['datetime'].dt.date.astype(str),
        'period': (df['datetime'].dt.hour * 4) + (df['datetime'].dt.minute // 15) + 1
        if resolucion == 'minutes15' else df['datetime'].dt.hour + 1,
        'wind_mwh': df['wind_mwh'].round(2),
        'solar_mwh': df['solar_mwh'].round(2),
        'renewable_total_mwh': df['renewable_total_mwh'].round(2),
    })
    df_final.reset_index(drop=True, inplace=True)

    return df_final


# --- Ejemplo de uso ---
if __name__ == "__main__":
    MI_TOKEN_ESIOS = "3c43bf8c3fe7eba0eb28b662437605b10c81910b1502e346f597627fa8d54557"

    periodos = [
        (datetime(2025, 6, 1, 0, 0), datetime(2025, 8, 31, 23, 59)),
        (datetime(2025, 12, 1, 0, 0), datetime(2026, 2, 28, 23, 59)),
    ]
    resolucion = 'hour'

    for fecha_inicio, fecha_fin in periodos:
        df_gen = obtener_generacion_renovable(MI_TOKEN_ESIOS, fecha_inicio, fecha_fin, resolucion)

        if df_gen is not None and not df_gen.empty:
            print(df_gen.head())
            print(f"Total de horas recuperadas: {len(df_gen)}")
            nombre = (
                f"./data/generacion_renovable_{fecha_inicio.strftime('%Y-%m-%d')}"
                f"_{fecha_fin.strftime('%Y-%m-%d')}_{resolucion}.csv"
            )
            df_gen.to_csv(nombre, index=False)
            print(f"Generación guardada en {nombre}\n")
