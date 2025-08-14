"""
Получение исторических данных с Bybit (БЕЗ ПРИВАТНЫХ КЛЮЧЕЙ)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BybitDataFetcher:
    """Класс для получения исторических данных с Bybit (только публичные данные)"""

    def __init__(self, testnet: bool = False):
        """
        Инициализация клиента Bybit (БЕЗ API КЛЮЧЕЙ)

        Args:
            testnet: Использовать тестовую сеть
        """
        self.session = HTTP(testnet=testnet)
        self.rate_limit_delay = 0.1  # Задержка между запросами

    def _convert_timeframe(self, timeframe: str) -> int:
        """Конвертация таймфрейма в минуты для Bybit API"""
        timeframe_map = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440, "1w": 10080
        }
        return timeframe_map.get(timeframe, 15)

    def _datetime_to_timestamp(self, dt_str: str) -> int:
        """Конвертация даты в timestamp для Bybit API"""
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

    def get_historical_klines(self, symbol: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Получение исторических свечей (БЕЗ АУТЕНТИФИКАЦИИ)

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            timeframe: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Дата начала (YYYY-MM-DD)
            end_date: Дата окончания (YYYY-MM-DD)

        Returns:
            DataFrame с OHLCV данными
        """
        logger.info(f"Получение данных для {symbol} {timeframe} с {start_date} по {end_date}")

        interval = self._convert_timeframe(timeframe)
        start_ts = self._datetime_to_timestamp(start_date)
        end_ts = self._datetime_to_timestamp(end_date)

        all_data = []
        current_start = start_ts

        # Bybit возвращает максимум 1000 свечей за запрос
        max_candles = 1000
        interval_ms = interval * 60 * 1000

        while current_start < end_ts:
            current_end = min(current_start + (max_candles * interval_ms), end_ts)

            try:
                response = self.session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=current_start,
                    end=current_end
                )

                if response['retCode'] == 0 and response['result']['list']:
                    klines = response['result']['list']
                    all_data.extend(klines)
                    logger.info(f"Получено {len(klines)} свечей для {symbol}")
                else:
                    logger.warning(f"Нет данных для {symbol} в периоде {current_start}-{current_end}")

                current_start = current_end
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Ошибка получения данных для {symbol}: {e}")
                break

        if not all_data:
            logger.error(f"Не удалось получить данные для {symbol}")
            return pd.DataFrame()

        # Конвертация в DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Обработка данных
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        df = df.sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)

        logger.info(f"Итого получено {len(df)} свечей для {symbol} {timeframe}")
        return df

    def get_multiple_symbols(self, symbols: List[str], timeframe: str,
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Получение данных для нескольких символов

        Args:
            symbols: Список торговых пар
            timeframe: Таймфрейм
            start_date: Дата начала
            end_date: Дата окончания

        Returns:
            Словарь {symbol: DataFrame}
        """
        data = {}

        for symbol in symbols:
            try:
                df = self.get_historical_klines(symbol, timeframe, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
                    logger.info(f"Успешно получены данные для {symbol}")
                else:
                    logger.warning(f"Пустые данные для {symbol}")

                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Ошибка получения данных для {symbol}: {e}")
                continue

        return data

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Валидация полученных данных

        Args:
            df: DataFrame с данными
            symbol: Символ для логирования

        Returns:
            True если данные валидны
        """
        if df.empty:
            logger.error(f"Пустые данные для {symbol}")
            return False

        # Проверка на пропуски
        if df.isnull().any().any():
            logger.warning(f"Найдены пропуски в данных для {symbol}")
            df.fillna(method='ffill', inplace=True)

        # Проверка на аномальные значения
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                logger.error(f"Найдены нулевые или отрицательные цены для {symbol}")
                return False

        # Проверка логики OHLC
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_ohlc.any():
            logger.error(f"Найдены некорректные OHLC данные для {symbol}")
            return False

        logger.info(f"Данные для {symbol} прошли валидацию")
        return True
