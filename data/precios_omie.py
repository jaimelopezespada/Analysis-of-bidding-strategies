import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

MADRID_TZ = ZoneInfo('Europe/Madrid')


def normalizar_dias_dst(df, columnas_valor, periodos_dia):
    """Deja todos los dias con exactamente `periodos_dia` periodos.

    En el cambio de hora de octubre el dia local tiene 25 horas (la hora
    02:00 ocurre dos veces): se promedia la hora repetida. En el de marzo
    tiene 23 (las 02:00 no existen): se rellena por interpolacion lineal.
    Necesario porque el pipeline (src/bidding/prices.py) exige el mismo
    numero de periodos en todos los dias.
    """
    agg = {c: 'mean' for c in columnas_valor}
    if 'date_aware' in df.columns:
        agg['date_aware'] = 'first'
    df = df.groupby(['date', 'period'], as_index=False).agg(agg)

    completo = pd.MultiIndex.from_product(
        [df['date'].unique(), range(1, periodos_dia + 1)],
        names=['date', 'period'],
    )
    df = df.set_index(['date', 'period']).reindex(completo)
    for c in columnas_valor:
        df[c] = df[c].interpolate(limit=4).round(2)
    return df.reset_index()

def obtener_precios_mercado_diario(api_key, start_date, end_date, resolucion):
    """
    Obtiene los precios horarios del mercado diario (OMIE) desde ESIOS.

    Parámetros:
    - api_key (str): Tu token personal de la API de ESIOS.
    - start_date (datetime): Fecha y hora de inicio en hora local de Madrid
      (si no lleva tzinfo, se asume Europe/Madrid).
    - end_date (datetime): Fecha y hora de fin en hora local de Madrid
      (si no lleva tzinfo, se asume Europe/Madrid).
    """

    # Interpretamos las fechas como hora local de Madrid y las convertimos
    # a UTC para la API. Así el rango pedido siempre coincide con días
    # locales completos (period 1 a 24), sin preocuparse del cambio de
    # horario de verano/invierno.
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

    # El indicador 600 corresponde al "Precio mercado diario" en ESIOS
    indicator_id = 600
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"

    # ESIOS requiere que el token se pase en este formato específico
    headers = {
        'Accept': 'application/json; application/vnd.esios-api-v1+json',
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # ESIOS agrega este indicador por SUMA (no por media) al truncar de 15
    # minutos a hora/día. Desde que el mercado pasó a resolución de 15
    # minutos, pedir time_trunc='hour' directamente devuelve la suma de los
    # 4 cuartos de hora en vez del precio medio horario. Para evitarlo,
    # siempre pedimos el detalle a 15 minutos y, si hace falta 'hour',
    # promediamos nosotros mismos más abajo.
    api_resolucion = 'minutes15' if resolucion == 'hour' else resolucion

    # Formato ISO 8601 que espera la API
    params = {
        'start_date': start_date_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'end_date': end_date_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'time_trunc': api_resolucion
    }

    print(f"Consultando datos desde {start_date} hasta {end_date} (hora Madrid)...")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extraer la lista de valores
        if 'indicator' in data and 'values' in data['indicator']:
            valores = data['indicator']['values']
            df = pd.DataFrame(valores)

            if df.empty:
                print("La consulta fue exitosa, pero no hay datos para este periodo.")
                return pd.DataFrame()

            # El indicador 600 incluye precios de España y Portugal (geo_id 3 y 4).
            # Nos quedamos solo con España.
            if 'geo_id' in df.columns:
                df = df[df['geo_id'] == 3]
            elif 'geo_name' in df.columns:
                df = df[df['geo_name'] == 'España']

            # Limpieza y formateo del DataFrame. utc=True es imprescindible
            # en rangos que cruzan el cambio de hora (oct/mar): la respuesta
            # mezcla offsets +02:00 y +01:00 y sin ello pandas devuelve una
            # columna object con la que .dt.tz_convert falla.
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

            # Convertir a la zona horaria de Madrid
            df['datetime'] = df['datetime'].dt.tz_convert('Europe/Madrid')
            df.sort_values('datetime', inplace=True)

            # Si se pidió resolución horaria pero hemos consultado a 15
            # minutos (ver comentario más arriba), promediamos aquí los
            # cuartos de cada hora en vez de dejar que ESIOS los sume.
            if resolucion == 'hour' and api_resolucion == 'minutes15':
                # El floor se hace en UTC porque sobre hora local falla en
                # el cambio horario de octubre (las 02:00 son ambiguas).
                hora = (
                    df['datetime'].dt.tz_convert('UTC')
                    .dt.floor('h')
                    .dt.tz_convert(MADRID_TZ)
                )
                df = df.groupby(hora)['value'].mean().reset_index()

            # Formato final: date, period (hora 1-24), time_aware (datetime con tz) y price (EUR/MWh)
            df_final = pd.DataFrame({
                'date_aware': df['datetime'],
                'date': df['datetime'].dt.date.astype(str),
                'period': (df['datetime'].dt.hour * 4) + (df['datetime'].dt.minute // 15) + 1 if resolucion == 'minutes15' else df['datetime'].dt.hour + 1,
                'price': df['value'].round(2),
            })
            df_final.reset_index(drop=True, inplace=True)

            if resolucion in ('hour', 'minutes15'):
                periodos_dia = 24 if resolucion == 'hour' else 96
                df_final = normalizar_dias_dst(df_final, ['price'], periodos_dia)

            return df_final
        else:
            print("Estructura JSON inesperada.")
            return None
    else:
        print(f"Error en la petición: {response.status_code}")
        print(response.text)
        return None

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Necesitas solicitar un token gratuito escribiendo a consultasios@ree.es
    MI_TOKEN_ESIOS = "3c43bf8c3fe7eba0eb28b662437605b10c81910b1502e346f597627fa8d54557"
    
    # Definir el periodo temporal en hora local de Madrid (días completos: 00:00 a 23:59)
    fecha_inicio = datetime(2025, 8, 1, 0, 0)
    fecha_fin = datetime(2025, 10, 31, 23, 59)

    # fecha_inicio = datetime(2025, 12, 1, 0, 0)
    # fecha_fin = datetime(2026, 2, 28, 23, 59)

    resolucion = 'hour'  # Puede ser 'minutes15', 'hour' o 'day'
    # Llamada a la función
    df_precios = obtener_precios_mercado_diario(MI_TOKEN_ESIOS, fecha_inicio, fecha_fin, resolucion)
    
    if df_precios is not None and not df_precios.empty:
        print("\nPrimeras horas extraídas:")
        print(df_precios.head())
        print(f"\nTotal de horas recuperadas: {len(df_precios)}")
        df_precios.to_csv(f"./data/precios_omie_{fecha_inicio.strftime('%Y-%m-%d')}_{fecha_fin.strftime('%Y-%m-%d')}_{resolucion}.csv", index=False)
        print("Precios guardados")