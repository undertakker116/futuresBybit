"""
Демо версия получения данных (локальные CSV файлы)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoDataFetcher:
    """Класс для получения демо данных из локальных CSV файлов"""

    def __init__(self):
        """Инициализация демо data fetcher"""
        self.data_dir = "demo_data"
        self.rate_limit_delay = 0.01  # Минимальная задержка для демо

    def get_historical_klines(self, symbol: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Получение исторических свечей из локальных файлов

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            timeframe: Таймфрейм (15m, 1h)
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)

        Returns:
            DataFrame с колонками: open, high, low, close, volume
        """
        
        logger.info(f"Загрузка демо данных для {symbol} {timeframe} с {start_date} по {end_date}")
        
        try:
            # Определяем файл для загрузки
            filename = f"{symbol}_{timeframe}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Файл {filepath} не найден")
                return pd.DataFrame()
            
            # Загружаем данные
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Фильтруем по датам
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            logger.info(f"Загружено {len(df)} свечей для {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке демо данных для {symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_symbols(self, symbols: List[str], timeframe: str,
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Получение данных для нескольких символов одновременно

        Args:
            symbols: Список торговых пар
            timeframe: Таймфрейм
            start_date: Начальная дата
            end_date: Конечная дата

        Returns:
            Словарь с данными для каждого символа
        """
        
        data = {}
        
        for symbol in symbols:
            df = self.get_historical_klines(symbol, timeframe, start_date, end_date)
            if not df.empty:
                data[symbol] = df
            else:
                logger.warning(f"Пустые данные для {symbol}")
        
        return data

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Валидация загруженных данных

        Args:
            df: DataFrame с данными
            symbol: Символ для логирования

        Returns:
            True если данные валидны, False иначе
        """
        
        if df.empty:
            logger.warning(f"Пустые данные для {symbol}")
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Отсутствуют колонки для {symbol}: {missing_columns}")
            return False
        
        # Проверяем на NaN значения
        if df[required_columns].isnull().any().any():
            logger.warning(f"Найдены NaN значения в данных для {symbol}")
            # Заполняем NaN значения методом forward fill
            df[required_columns] = df[required_columns].fillna(method='ffill')
        
        # Проверяем логику OHLC
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            logger.warning(f"Найдены некорректные OHLC данные для {symbol}: {invalid_rows.sum()} строк")
            # Исправляем некорректные данные
            df.loc[invalid_rows, 'high'] = df.loc[invalid_rows, ['open', 'close']].max(axis=1)
            df.loc[invalid_rows, 'low'] = df.loc[invalid_rows, ['open', 'close']].min(axis=1)
        
        logger.info(f"Данные для {symbol} прошли валидацию")
        return True