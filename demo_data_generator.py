import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def generate_realistic_demo_data(symbol: str, start_date: str, end_date: str, timeframe: str = "15m") -> pd.DataFrame:
    """Генерация реалистичных демо данных для тестирования"""
    
    # Преобразуем даты
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Определяем интервал для таймфрейма
    if timeframe == "15m":
        freq = "15min"
        intervals_per_day = 96  # 24 * 4
    elif timeframe == "1h":
        freq = "1h"
        intervals_per_day = 24
    else:
        freq = "15min"
        intervals_per_day = 96
    
    # Создаем временной ряд
    timestamps = pd.date_range(start=start, end=end, freq=freq)
    
    # Базовые параметры для разных символов
    if "FART" in symbol:
        base_price = 1.2
        volatility = 0.15  # Высокая волатильность для мем-коина
        trend_strength = 0.02
    elif "SOL" in symbol:
        base_price = 160.0
        volatility = 0.08  # Умеренная волатильность
        trend_strength = 0.01
    else:
        base_price = 100.0
        volatility = 0.10
        trend_strength = 0.015
    
    # Генерируем случайные движения с трендовыми компонентами
    np.random.seed(42)  # Для воспроизводимости
    
    n_periods = len(timestamps)
    
    # Создаем трендовые волны разной частоты
    trend_long = np.sin(np.linspace(0, 4 * np.pi, n_periods)) * trend_strength * base_price
    trend_medium = np.sin(np.linspace(0, 12 * np.pi, n_periods)) * trend_strength * base_price * 0.5
    trend_short = np.sin(np.linspace(0, 24 * np.pi, n_periods)) * trend_strength * base_price * 0.3
    
    # Случайные движения
    random_moves = np.random.normal(0, volatility * base_price * 0.01, n_periods)
    
    # Комбинируем все компоненты
    price_changes = trend_long + trend_medium + trend_short + random_moves
    
    # Создаем цены закрытия
    close_prices = [base_price]
    for change in price_changes[1:]:
        new_price = close_prices[-1] + change
        close_prices.append(max(new_price, base_price * 0.1))  # Предотвращаем отрицательные цены
    
    close_prices = np.array(close_prices)
    
    # Генерируем OHLV данные
    data = []
    for i, timestamp in enumerate(timestamps):
        close = close_prices[i]
        
        # Генерируем внутрибарную волатильность
        bar_volatility = np.random.uniform(0.005, 0.025) * close
        
        # High и Low
        high = close + np.random.uniform(0, bar_volatility)
        low = close - np.random.uniform(0, bar_volatility)
        
        # Open (на основе предыдущего close с небольшим гэпом)
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.002 * close)
            open_price = close_prices[i-1] + gap
            
        # Убеждаемся, что OHLC логически корректны
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume (реалистичный объем)
        base_volume = np.random.uniform(10000, 100000)
        volume_multiplier = 1 + abs(close - open_price) / close * 10  # Больше объема при больших движениях
        volume = base_volume * volume_multiplier
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def save_demo_data():
    """Сохранение демо данных для тестирования"""
    
    symbols = ["FARTCOINUSDT", "SOLUSDT"]
    start_date = "2024-05-01"
    end_date = "2024-08-13"
    
    # Создаем директорию для данных
    os.makedirs("demo_data", exist_ok=True)
    
    data = {}
    
    for symbol in symbols:
        print(f"Генерация демо данных для {symbol}...")
        
        # 15-минутные данные
        df_15m = generate_realistic_demo_data(symbol, start_date, end_date, "15m")
        data[symbol] = df_15m
        
        # Часовые данные (для MACD)
        df_1h = generate_realistic_demo_data(symbol, start_date, end_date, "1h")
        
        # Сохраняем в CSV
        df_15m.to_csv(f"demo_data/{symbol}_15m.csv")
        df_1h.to_csv(f"demo_data/{symbol}_1h.csv")
        
        print(f"Создано {len(df_15m)} 15-минутных свечей и {len(df_1h)} часовых свечей для {symbol}")
    
    print("Демо данные сохранены в папку demo_data/")
    return data

if __name__ == "__main__":
    save_demo_data()