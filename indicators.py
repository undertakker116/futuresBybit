import numpy as np
import pandas as pd
from typing import Dict, Any


class TechnicalIndicators:
    """Оптимизированная Parabolic SAR стратегия с несколькими подходами"""

    @staticmethod
    def _calculate_sar(high: np.ndarray, low: np.ndarray, acceleration: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        """Собственная реализация Parabolic SAR"""
        length = len(high)
        sar = np.full(length, np.nan)
        af = acceleration
        ep = 0.0
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        if length < 2:
            return sar
            
        # Инициализация
        sar[0] = low[0]
        ep = high[0]
        
        for i in range(1, length):
            # Расчет SAR
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            
            # Проверка разворота тренда
            if trend == 1:  # Восходящий тренд
                if low[i] <= sar[i]:
                    # Разворот в нисходящий тренд
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    # Продолжение восходящего тренда
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
                    # Корректировка SAR для восходящего тренда
                    sar[i] = min(sar[i], low[i-1])
                    if i > 1:
                        sar[i] = min(sar[i], low[i-2])
            else:  # Нисходящий тренд
                if high[i] >= sar[i]:
                    # Разворот в восходящий тренд
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    # Продолжение нисходящего тренда
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
                    # Корректировка SAR для нисходящего тренда
                    sar[i] = max(sar[i], high[i-1])
                    if i > 1:
                        sar[i] = max(sar[i], high[i-2])
        
        return sar

    @staticmethod
    def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Собственная реализация Average True Range"""
        if len(high) < period:
            return np.full(len(high), np.nan)
            
        # Расчет True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Первое значение
        
        # Расчет ATR как скользящее среднее TR
        atr = np.full(len(tr), np.nan)
        atr[period-1:] = pd.Series(tr).rolling(window=period).mean()[period-1:]
        
        return atr

    @staticmethod
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """Собственная реализация Exponential Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
            
        alpha = 2.0 / (period + 1)
        ema = np.full(len(data), np.nan)
        
        # Первое значение - простое среднее
        ema[period-1] = np.mean(data[:period])
        
        # Расчет EMA
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema

    @staticmethod
    def _calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Собственная реализация Relative Strength Index"""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
            
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Первые значения как простое среднее
        avg_gain = np.full(len(data), np.nan)
        avg_loss = np.full(len(data), np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Сглаженные средние
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def _calculate_macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Собственная реализация MACD"""
        if len(data) < slow + signal:
            length = len(data)
            return np.full(length, np.nan), np.full(length, np.nan), np.full(length, np.nan)
            
        ema_fast = TechnicalIndicators._calculate_ema(data, fast)
        ema_slow = TechnicalIndicators._calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Убираем NaN для расчета сигнальной линии
        valid_macd = macd_line[~np.isnan(macd_line)]
        if len(valid_macd) < signal:
            return macd_line, np.full(len(macd_line), np.nan), np.full(len(macd_line), np.nan)
        
        signal_line_values = TechnicalIndicators._calculate_ema(valid_macd, signal)
        
        # Создаем полную сигнальную линию
        full_signal = np.full(len(macd_line), np.nan)
        valid_start = np.where(~np.isnan(macd_line))[0]
        
        if len(valid_start) > 0 and len(signal_line_values[~np.isnan(signal_line_values)]) > 0:
            signal_start_idx = valid_start[0] + signal - 1
            signal_values = signal_line_values[~np.isnan(signal_line_values)]
            
            if signal_start_idx < len(full_signal):
                end_idx = min(signal_start_idx + len(signal_values), len(full_signal))
                full_signal[signal_start_idx:end_idx] = signal_values[:end_idx-signal_start_idx]
        
        histogram = macd_line - full_signal
        
        return macd_line, full_signal, histogram

    @staticmethod
    def _calculate_bbands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """Собственная реализация Bollinger Bands"""
        if len(data) < period:
            length = len(data)
            return np.full(length, np.nan), np.full(length, np.nan), np.full(length, np.nan)
            
        # Простое скользящее среднее
        sma = np.full(len(data), np.nan)
        std = np.full(len(data), np.nan)
        
        for i in range(period-1, len(data)):
            window = data[i-period+1:i+1]
            sma[i] = np.mean(window)
            std[i] = np.std(window, ddof=0)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower

    @staticmethod
    def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Собственная реализация Simple Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
            
        sma = np.full(len(data), np.nan)
        
        for i in range(period-1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
            
        return sma

    @staticmethod
    def parabolic_sar_strategy(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Оптимизированная SAR стратегия с пятью режимами для максимального винрейта
        """
        high = df['high'].astype(np.float64).values
        low = df['low'].astype(np.float64).values
        close = df['close'].astype(np.float64).values
        volume = df['volume'].astype(np.float64).values if 'volume' in df.columns else None

        strategy_mode = config.get('strategy_mode', 'optimized')

        if strategy_mode == 'classic':
            sar = TechnicalIndicators._calculate_sar(high, low, acceleration=0.02, maximum=0.2)
        elif strategy_mode == 'contrarian':
            sar = TechnicalIndicators._calculate_sar(high, low, acceleration=0.02, maximum=0.2)
        elif strategy_mode == 'trend_following':
            sar = TechnicalIndicators._calculate_sar(high, low, acceleration=0.01, maximum=0.12)
        elif strategy_mode == 'high_winrate':
            sar = TechnicalIndicators._calculate_sar(high, low, acceleration=0.008, maximum=0.08)
        else:  # optimized
            sar = TechnicalIndicators._calculate_sar(high, low, acceleration=0.01, maximum=0.1)

        # Расчет дополнительных индикаторов для фильтрации
        atr = TechnicalIndicators._calculate_atr(high, low, close, period=14)
        ema20 = TechnicalIndicators._calculate_ema(close, period=20)
        ema50 = TechnicalIndicators._calculate_ema(close, period=50)
        ema200 = TechnicalIndicators._calculate_ema(close, period=200)  # Долгосрочный тренд
        rsi = TechnicalIndicators._calculate_rsi(close, period=14)
        macd, macd_signal, macd_hist = TechnicalIndicators._calculate_macd(close)  # Дополнительное подтверждение
        bb_upper, bb_middle, bb_lower = TechnicalIndicators._calculate_bbands(close)  # Болинджер для волатильности

        if volume is not None:
            volume_sma = TechnicalIndicators._calculate_sma(volume, period=20)
            volume_ratio = volume / volume_sma
        else:
            volume_ratio = np.ones_like(close)

        # Определение направления SAR тренда
        sar_trend_up = sar < close
        sar_trend_down = sar > close

        sar_trend_up_series = pd.Series(sar_trend_up, index=df.index)
        sar_trend_down_series = pd.Series(sar_trend_down, index=df.index)

        if strategy_mode == 'contrarian':
            # Противоположная логика для контрарианской стратегии
            sar_long_signal = sar_trend_down_series & (
                ~sar_trend_down_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            sar_short_signal = sar_trend_up_series & (
                ~sar_trend_up_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            long_exit = sar_trend_up_series & (
                ~sar_trend_up_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            short_exit = sar_trend_down_series & (
                ~sar_trend_down_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
        else:
            sar_long_signal = sar_trend_up_series & (
                ~sar_trend_up_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            sar_short_signal = sar_trend_down_series & (
                ~sar_trend_down_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            long_exit = sar_trend_down_series & (
                ~sar_trend_down_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))
            short_exit = sar_trend_up_series & (
                ~sar_trend_up_series.shift(1).infer_objects(copy=False).fillna(False).astype(bool))

        if strategy_mode in ['optimized', 'trend_following', 'high_winrate']:
            # Основной тренд (EMA50 и EMA200)
            strong_uptrend = (close > ema50) & (ema50 > ema200) & (close > ema200)
            strong_downtrend = (close < ema50) & (ema50 < ema200) & (close < ema200)

            # Волатильность и сила движения
            min_volatility = 0.015 if strategy_mode == 'high_winrate' else 0.008
            volatility_filter = (atr / close) > min_volatility

            # RSI в нормальной зоне
            rsi_bounds = (35, 65) if strategy_mode == 'high_winrate' else (30, 70)
            rsi_normal = (rsi > rsi_bounds[0]) & (rsi < rsi_bounds[1])

            # MACD подтверждение
            macd_bullish = macd > macd_signal
            macd_bearish = macd < macd_signal

            # Объемное подтверждение
            volume_confirmation = volume_ratio > 1.2 if strategy_mode == 'high_winrate' else volume_ratio > 1.0

            # Пробой Болинджера (для high_winrate режима)
            if strategy_mode == 'high_winrate':
                bb_breakout_up = close > bb_upper
                bb_breakout_down = close < bb_lower
                price_momentum_up = close > ema20 * 1.01  # Цена выше EMA20 на 1%
                price_momentum_down = close < ema20 * 0.99  # Цена ниже EMA20 на 1%
            else:
                bb_breakout_up = True
                bb_breakout_down = True
                price_momentum_up = close > ema20
                price_momentum_down = close < ema20

            # Комбинированные фильтры для максимального качества
            long_entry = (sar_long_signal &
                          strong_uptrend &
                          price_momentum_up &
                          volatility_filter &
                          rsi_normal &
                          macd_bullish &
                          volume_confirmation)

            short_entry = (sar_short_signal &
                           strong_downtrend &
                           price_momentum_down &
                           volatility_filter &
                           rsi_normal &
                           macd_bearish &
                           volume_confirmation)

            if strategy_mode == 'high_winrate':
                long_entry = long_entry & bb_breakout_up
                short_entry = short_entry & bb_breakout_down
        else:
            long_entry = sar_long_signal
            short_entry = sar_short_signal

        # Результат
        result = df.copy()
        result['psar'] = sar
        result['psar_trend_up'] = sar_trend_up
        result['psar_trend_down'] = sar_trend_down
        result['atr'] = atr
        result['ema20'] = ema20
        result['ema50'] = ema50
        result['ema200'] = ema200  # Добавил долгосрочный тренд
        result['rsi'] = rsi
        result['macd'] = macd  # Добавил MACD
        result['bb_upper'] = bb_upper  # Добавил Болинджер
        result['bb_lower'] = bb_lower
        result['volume_ratio'] = volume_ratio  # Добавил объемный анализ

        result['sar_long_entry'] = long_entry.values
        result['sar_short_entry'] = short_entry.values
        result['sar_long_exit'] = long_exit.values
        result['sar_short_exit'] = short_exit.values

        return result

    @staticmethod
    def calculate_macd_trend(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, Any]:
        """
        Расчет MACD индикатора для определения тренда на часовых данных
        Возвращает MACD линию, сигнальную линию и гистограмму
        """
        close = df['close'].astype(np.float64).values
        
        # Расчет MACD
        macd_line, macd_signal, macd_histogram = TechnicalIndicators._calculate_macd(close, 
                                                                                     fast=fast_period,
                                                                                     slow=slow_period, 
                                                                                     signal=signal_period)
        
        # Определение направления тренда
        macd_trend_bullish = (macd_line > macd_signal) & (macd_histogram > 0)
        macd_trend_bearish = (macd_line < macd_signal) & (macd_histogram < 0)
        
        # Сила тренда (увеличивающаяся гистограмма)
        macd_strength_increasing = np.diff(macd_histogram, prepend=np.nan) > 0
        
        return {
            'macd_line': pd.Series(macd_line, index=df.index),
            'macd_signal': pd.Series(macd_signal, index=df.index),
            'macd_histogram': pd.Series(macd_histogram, index=df.index),
            'macd_trend_bullish': pd.Series(macd_trend_bullish, index=df.index),
            'macd_trend_bearish': pd.Series(macd_trend_bearish, index=df.index),
            'macd_strength_increasing': pd.Series(macd_strength_increasing, index=df.index)
        }
