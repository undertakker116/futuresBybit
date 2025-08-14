import numpy as np
import pandas as pd
from typing import Dict, Any
import talib


class TechnicalIndicators:
    """Оптимизированная Parabolic SAR стратегия с несколькими подходами"""

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
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        elif strategy_mode == 'contrarian':
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        elif strategy_mode == 'trend_following':
            sar = talib.SAR(high, low, acceleration=0.01, maximum=0.12)
        elif strategy_mode == 'high_winrate':
            sar = talib.SAR(high, low, acceleration=0.008, maximum=0.08)
        else:  # optimized
            sar = talib.SAR(high, low, acceleration=0.01, maximum=0.1)

        # Расчет дополнительных индикаторов для фильтрации
        atr = talib.ATR(high, low, close, timeperiod=14)
        ema20 = talib.EMA(close, timeperiod=20)
        ema50 = talib.EMA(close, timeperiod=50)
        ema200 = talib.EMA(close, timeperiod=200)  # Долгосрочный тренд
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)  # Дополнительное подтверждение
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)  # Болинджер для волатильности

        if volume is not None:
            volume_sma = talib.SMA(volume, timeperiod=20)
            volume_ratio = volume / volume_sma
        else:
            volume_ratio = np.ones_like(close)

        # Определение направления SAR тренда
        sar_trend_up = sar < close
        sar_trend_down = sar > close

        sar_trend_up_series = pd.Series(sar_trend_up, index=df.index)
        sar_trend_down_series = pd.Series(sar_trend_down, index=df.index)

        if strategy_mode == 'contrarian':
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
