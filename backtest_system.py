"""
Система бэктестинга для чистой Parabolic SAR стратегии
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os
from dataclasses import dataclass, asdict
import json

from config import *
from data_fetcher import BybitDataFetcher
from indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Структура для хранения информации о сделке"""
    entry_time: str
    exit_time: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    duration_hours: float

class SARBacktestEngine:
    """Движок бэктестинга для чистой Parabolic SAR стратегии"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_fetcher = BybitDataFetcher()
        self.indicators = TechnicalIndicators()

        self.trades: List[Trade] = []
        self.current_position = None  # Текущая позиция (только одна)
        self.equity_curve = []
        self.current_capital = config['initial_capital']
        self.peak_capital = config['initial_capital']
        self.max_drawdown = 0.0
        self.last_trade_time = None

        # Кэш данных старшего ТФ (1h)
        self.htf_timeframe = '1h'
        self.htf_data: Dict[str, pd.DataFrame] = {}

        # Создание директории для результатов
        if OUTPUT_CONFIG['save_results']:
            os.makedirs(OUTPUT_CONFIG['results_dir'], exist_ok=True)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка исторических данных"""
        logger.info("Загрузка исторических данных...")

        try:
            data = self.data_fetcher.get_multiple_symbols(
                SYMBOLS,
                PRIMARY_TIMEFRAME,
                START_DATE,
                END_DATE
            )

            # Валидация данных
            validated_data = {}
            for symbol, df in data.items():
                if self.data_fetcher.validate_data(df, symbol):
                    validated_data[symbol] = df

            logger.info(f"Загружены данные для {len(validated_data)} символов")

            # Дополнительно загружаем 1h для тренд-фильтра MACD
            logger.info(f"Загрузка данных старшего ТФ ({self.htf_timeframe}) для MACD тренда...")
            htf = self.data_fetcher.get_multiple_symbols(
                SYMBOLS,
                self.htf_timeframe,
                START_DATE,
                END_DATE
            )
            # Предобработка: расчет MACD на 1h и подготовка к последующему merge_asof
            for symbol, df1h in htf.items():
                if not df1h.empty:
                    macd, macd_signal, macd_hist = self._compute_macd(df1h['close'].astype(np.float64).values)
                    df1h = df1h.copy()
                    df1h['macd_1h'] = macd
                    df1h['macd_signal_1h'] = macd_signal
                    df1h['macd_hist_1h'] = macd_hist
                    df1h['close_1h'] = df1h['close'].astype(np.float64)
                    # Оставляем только нужные колонки для объединения
                    self.htf_data[symbol] = df1h[['macd_1h', 'macd_signal_1h', 'macd_hist_1h', 'close_1h']].copy()

            return validated_data

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            return {}

    def _compute_macd(self, close_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вспомогательный метод для расчета MACD (использует talib, если доступен)."""
        try:
            import talib
            macd, macd_signal, macd_hist = talib.MACD(close_values)
            return macd, macd_signal, macd_hist
        except Exception:
            # Фоллбэк реализация MACD (12,26,9)
            short_ema = pd.Series(close_values).ewm(span=12, adjust=False).mean().values
            long_ema = pd.Series(close_values).ewm(span=26, adjust=False).mean().values
            macd = short_ema - long_ema
            macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist

    def _merge_htf_macd(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Обогащает df (на 15m) колонками MACD 1h c помощью merge_asof."""
        if symbol not in self.htf_data or self.htf_data[symbol].empty:
            return df
        df_15m = df.copy().sort_index()
        df_15m['ts'] = df_15m.index
        df1h = self.htf_data[symbol].copy().sort_index()
        df1h['ts'] = df1h.index
        merged = pd.merge_asof(
            df_15m.reset_index(drop=True).sort_values('ts'),
            df1h.reset_index(drop=True).sort_values('ts'),
            on='ts',
            direction='backward'
        )
        merged.set_index('ts', inplace=True)
        # Сохраняем исходные OHLCV и добавленные 1h колонки
        for col in ['macd_1h', 'macd_signal_1h', 'macd_hist_1h', 'close_1h']:
            if col in merged.columns:
                df_15m[col] = merged[col].astype(np.float64)
        df_15m.index.name = df.index.name
        return df_15m

    def calculate_indicators(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет Parabolic SAR индикатора + добавление 1h MACD тренда"""

        try:
            df_enriched = self._merge_htf_macd(symbol, df)
            df_with_ind = self.indicators.parabolic_sar_strategy(df_enriched, PSAR_CONFIG)
            return df_with_ind

        except Exception as e:
            logger.error(f"Ошибка при расчете индикаторов: {e}")
            return df

    def calculate_position_size(self, price: float) -> float:
        """Расчет размера позиции"""
        risk_amount = self.current_capital * self.config['risk_per_trade'] / 100
        quantity = abs(risk_amount / price)  # Всегда положительное значение
        return quantity

    def process_sar_signals(self, data: Dict[str, pd.DataFrame]) -> None:
        """Обработка SAR сигналов и создание сделок"""
        logger.info("Обработка SAR сигналов...")

        for symbol, df in data.items():
            logger.info(f"Обработка сигналов для {symbol}...")

            for i in range(PSAR_CONFIG['warmup_periods'], len(df)):
                timestamp = df.index[i]
                row = df.iloc[i]

                # Обновляем капитал в кривой эквити только один раз за временную метку
                if symbol == list(data.keys())[0]:  # Только для первого символа
                    self.equity_curve.append({
                        'timestamp': timestamp,
                        'capital': self.current_capital,
                        'drawdown': (self.peak_capital - self.current_capital) / self.peak_capital * 100
                    })

                    # Обновляем пиковый капитал
                    if self.current_capital > self.peak_capital:
                        self.peak_capital = self.current_capital

                    # Обновляем максимальную просадку
                    current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
                    if current_drawdown > self.max_drawdown:
                        self.max_drawdown = current_drawdown

                # Обрабатываем сигнал для текущего символа
                self.process_sar_signal(symbol, row, timestamp, df)

    def process_sar_signal(self, symbol: str, row: pd.Series, timestamp, df: pd.DataFrame) -> None:
        """Обработка SAR сигнала для конкретного символа"""

        if self.current_position and self.current_position['symbol'] != symbol:
            return  # Пропускаем, если открыта позиция по другому символу

        # Если есть открытая позиция по этому символу
        if self.current_position and self.current_position['symbol'] == symbol:
            if self.current_position['side'] == 'long' and row['sar_long_exit']:
                self.close_position(row['close'], timestamp, 'sar_reversal', row)
            elif self.current_position['side'] == 'short' and row['sar_short_exit']:
                self.close_position(row['close'], timestamp, 'sar_reversal', row)

            # Проверяем стоп-лосс и тейк-профит как подстраховку
            elif self.current_position:
                self.check_stop_take_levels(row['close'], timestamp, row)

        # Если нет позиции, проверяем входы
        elif not self.current_position:

            # Вход в лонг по SAR
            if row['sar_long_entry']:
                self.open_position(symbol, 'long', row['close'], timestamp, row)

            # Вход в шорт по SAR
            elif row['sar_short_entry']:
                self.open_position(symbol, 'short', row['close'], timestamp, row)

    def open_position(self, symbol: str, side: str, entry_price: float,
                     timestamp, row: pd.Series) -> None:
        """Открытие позиции"""

        quantity = self.calculate_position_size(entry_price)

        # Рассчитываем стоп-лосс и тейк-профит как подстраховку
        atr = row.get('atr', entry_price * 0.015)  # Fallback к 1.5% если ATR нет

        if side == 'long':
            stop_loss = entry_price - (atr * self.config['stop_loss_atr_multiplier_long'])
            take_profit = entry_price + (atr * self.config['take_profit_atr_multiplier_long'])
        else:  # short
            stop_loss = entry_price + (atr * self.config['stop_loss_atr_multiplier_short'])
            take_profit = entry_price - (atr * self.config['take_profit_atr_multiplier_short'])

        # Сохраняем информацию об открытой позиции
        self.current_position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        self.last_trade_time = timestamp

        # Логируем ключевые индикаторы при входе
        if BACKTEST_CONFIG.get('detailed_logs', False):
            indicators_snapshot = self._format_indicators_snapshot(row)
            logger.info(
                f"Открыта {side} позиция по {symbol} по цене {entry_price:.4f} (SAR сигнал) | {indicators_snapshot}"
            )
        else:
            logger.info(f"Открыта {side} позиция по {symbol} по цене {entry_price:.4f} (SAR сигнал)")

    def _format_indicators_snapshot(self, row: pd.Series) -> str:
        """Формирует краткий срез значений индикаторов для логов"""
        def safe_float(v, nd=4):
            try:
                return f"{float(v):.{nd}f}"
            except Exception:
                return "nan"
        parts = []
        parts.append(f"psar={safe_float(row.get('psar'))}")
        parts.append(f"rsi={safe_float(row.get('rsi'))}")
        parts.append(f"atr={safe_float(row.get('atr'))}")
        parts.append(f"ema50={safe_float(row.get('ema50'))}")
        parts.append(f"ema200={safe_float(row.get('ema200'))}")
        if 'macd' in row:
            parts.append(f"macd15={safe_float(row.get('macd'))}")
        if 'macd_1h' in row:
            parts.append(f"macd1h={safe_float(row.get('macd_1h'))}")
        if 'macd_hist_1h' in row:
            parts.append(f"macd_hist1h={safe_float(row.get('macd_hist_1h'))}")
        if 'volume_ratio' in row:
            parts.append(f"vol_ratio={safe_float(row.get('volume_ratio'))}")
        return ", ".join(parts)

    def check_stop_take_levels(self, current_price: float, timestamp, row: Optional[pd.Series] = None) -> None:
        """Проверка стоп-лосса и тейк-профита как подстраховку"""

        if not self.current_position:
            return

        position = self.current_position

        # Проверяем стоп-лосс
        if position['side'] == 'long' and current_price <= position['stop_loss']:
            self.close_position(current_price, timestamp, 'stop_loss', row)
        elif position['side'] == 'short' and current_price >= position['stop_loss']:
            self.close_position(current_price, timestamp, 'stop_loss', row)

        # Проверяем тейк-профит
        elif position['side'] == 'long' and current_price >= position['take_profit']:
            self.close_position(current_price, timestamp, 'take_profit', row)
        elif position['side'] == 'short' and current_price <= position['take_profit']:
            self.close_position(current_price, timestamp, 'take_profit', row)

    def close_position(self, exit_price: float, timestamp, exit_reason: str, row: Optional[pd.Series] = None) -> None:
        """Закрытие позиции"""

        if not self.current_position:
            return

        position = self.current_position

        if exit_price <= 0 or position['entry_price'] <= 0:
            logger.error(f"Некорректные цены: вход={position['entry_price']}, выход={exit_price}")
            return

        # Рассчитываем P&L
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100
        else:  # short
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_percent = (position['entry_price'] - exit_price) / position['entry_price'] * 100

        # Обновляем капитал
        self.current_capital += pnl

        # Рассчитываем продолжительность сделки
        duration = (timestamp - position['entry_time']).total_seconds() / 3600  # в часах

        if abs(pnl_percent) > 1000:  # Если P&L больше 1000%, что-то не так
            logger.warning(f"Аномальный P&L: {pnl_percent:.2f}% для {position['symbol']}")

        # Создаем объект сделки
        trade = Trade(
            entry_time=position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            exit_time=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            symbol=position['symbol'],
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            duration_hours=duration
        )

        self.trades.append(trade)

        # Лог закрытия с индикаторами
        if BACKTEST_CONFIG.get('detailed_logs', False) and row is not None:
            indicators_snapshot = self._format_indicators_snapshot(row)
            logger.info(
                f"Закрыта {position['side']} позиция по {position['symbol']} "
                f"вход: {position['entry_price']:.4f}, выход: {exit_price:.4f}, "
                f"P&L: {pnl:.2f} ({pnl_percent:.2f}%), причина: {exit_reason} | {indicators_snapshot}"
            )
        else:
            logger.info(f"Закрыта {position['side']} позиция по {position['symbol']} "
                       f"вход: {position['entry_price']:.4f}, выход: {exit_price:.4f}, "
                       f"P&L: {pnl:.2f} ({pnl_percent:.2f}%), причина: {exit_reason}")

        # Очищаем текущую позицию
        self.current_position = None

    def save_trades_to_json(self) -> None:
        """Сохранение сделок в JSON файл"""
        if not OUTPUT_CONFIG['save_results']:
            return

        trades_data = [asdict(trade) for trade in self.trades]

        trades_path = os.path.join(OUTPUT_CONFIG['results_dir'], 'sar_trades.json')
        with open(trades_path, 'w', encoding='utf-8') as f:
            json.dump(trades_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Сделки сохранены в: {trades_path}")

    def calculate_trading_statistics(self) -> Dict:
        """Расчет торговой статистики"""
        if not self.trades:
            return {}

        # Основная статистика
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_percent = (self.current_capital - self.config['initial_capital']) / self.config['initial_capital'] * 100

        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')

        # Статистика по причинам выхода
        exit_stats = {}
        for exit_reason in ['sar_reversal', 'stop_loss', 'take_profit']:
            reason_trades = [t for t in self.trades if t.exit_reason == exit_reason]
            if reason_trades:
                reason_wins = [t for t in reason_trades if t.pnl > 0]
                exit_stats[exit_reason] = {
                    'count': len(reason_trades),
                    'win_rate': len(reason_wins) / len(reason_trades) * 100,
                    'avg_pnl': sum(t.pnl for t in reason_trades) / len(reason_trades)
                }

        # Статистика по направлениям
        long_trades = [t for t in self.trades if t.side == 'long']
        short_trades = [t for t in self.trades if t.side == 'short']

        direction_stats = {}
        if long_trades:
            long_wins = [t for t in long_trades if t.pnl > 0]
            direction_stats['long'] = {
                'count': len(long_trades),
                'win_rate': len(long_wins) / len(long_trades) * 100,
                'avg_pnl': sum(t.pnl for t in long_trades) / len(long_trades)
            }

        if short_trades:
            short_wins = [t for t in short_trades if t.pnl > 0]
            direction_stats['short'] = {
                'count': len(short_trades),
                'win_rate': len(short_wins) / len(short_trades) * 100,
                'avg_pnl': sum(t.pnl for t in short_trades) / len(short_trades)
            }

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'final_capital': self.current_capital,
            'exit_stats': exit_stats,
            'direction_stats': direction_stats
        }

    def run_backtest(self) -> Dict:
        """Запуск полного бэктестинга"""

        logger.info("Запуск SAR бэктестинга...")

        try:
            # Загрузка данных
            data = self.load_data()
            if not data:
                logger.error("Не удалось загрузить данные")
                return {}

            # Расчет индикаторов
            logger.info("Расчет Parabolic SAR...")
            for symbol in data:
                data[symbol] = self.calculate_indicators(symbol, data[symbol])

            # Обработка SAR сигналов
            self.process_sar_signals(data)

            # Закрываем позицию в конце периода если она открыта
            if self.current_position:
                # Находим последнюю цену для символа текущей позиции
                symbol = self.current_position['symbol']
                if symbol in data:
                    last_timestamp = data[symbol].index[-1]
                    last_price = data[symbol].loc[last_timestamp, 'close']
                    self.close_position(last_price, last_timestamp, 'end_of_period', data[symbol].iloc[-1])

            # Расчет статистики
            trading_stats = self.calculate_trading_statistics()

            # Сохранение сделок
            self.save_trades_to_json()

            # Генерация отчета
            self.generate_report(trading_stats)

            # Построение графиков
            if OUTPUT_CONFIG['plot_charts']:
                self.plot_results(data, trading_stats)

            return {'trading_stats': trading_stats}

        except Exception as e:
            logger.error(f"Ошибка при выполнении бэктестинга: {e}")
            return {}

    def generate_report(self, trading_stats: Dict) -> None:
        """Генерация отчета"""

        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ КЛАССИЧЕСКОЙ PARABOLIC SAR СТРАТЕГИИ")
        print("="*60)

        print(f"Период: {START_DATE} - {END_DATE}")
        print(f"Символы: {', '.join(SYMBOLS)}")
        print(f"Таймфрейм: {PRIMARY_TIMEFRAME}")
        print(f"Начальный капитал: ${self.config['initial_capital']:,.2f}")
        print(f"Прогрев SAR: {PSAR_CONFIG['warmup_periods']} свечей")
        print("Стратегия: Классическая SAR + мягкий тренд-фильтр MACD (1h)")

        if trading_stats:
            print(f"\nТОРГОВАЯ СТАТИСТИКА:")
            print(f"Всего сделок: {trading_stats['total_trades']}")
            print(f"Прибыльных: {trading_stats['winning_trades']} ({trading_stats['win_rate']:.1f}%)")
            print(f"Убыточных: {trading_stats['losing_trades']}")

            if trading_stats['win_rate'] >= 60:
                print("🎯 Отличный результат! Винрейт выше 60%")
            elif trading_stats['win_rate'] >= 50:
                print("✅ Хороший результат. Винрейт выше 50%")
            elif trading_stats['win_rate'] >= 40:
                print("⚠️ Средний результат. Требует оптимизации")
            else:
                print("📉 Требует улучшения. Винрейт ниже 40%")

            print(f"Общая прибыль: ${trading_stats['total_pnl']:,.2f} ({trading_stats['total_pnl_percent']:.2f}%)")
            print(f"Итоговый капитал: ${trading_stats['final_capital']:,.2f}")
            print(f"Средняя прибыльная сделка: ${trading_stats['avg_win']:,.2f}")
            print(f"Средняя убыточная сделка: ${trading_stats['avg_loss']:,.2f}")
            print(f"Профит-фактор: {trading_stats['profit_factor']:.2f}")
            print(f"Максимальная просадка: {trading_stats['max_drawdown']:.2f}%")

            # Статистика по причинам выхода
            if 'exit_stats' in trading_stats:
                print(f"\nСТАТИСТИКА ПО ПРИЧИНАМ ВЫХОДА:")
                for exit_reason, stats in trading_stats['exit_stats'].items():
                    print(f"{exit_reason.replace('_', ' ').upper()}: {stats['count']} сделок, "
                          f"винрейт {stats['win_rate']:.1f}%, средняя прибыль ${stats['avg_pnl']:.2f}")

            # Статистика по направлениям
            if 'direction_stats' in trading_stats:
                print(f"\nСТАТИСТИКА ПО НАПРАВЛЕНИЯМ:")
                for direction, stats in trading_stats['direction_stats'].items():
                    print(f"{direction.upper()}: {stats['count']} сделок, "
                          f"винрейт {stats['win_rate']:.1f}%, средняя прибыль ${stats['avg_pnl']:.2f}")

    def plot_results(self, data: Dict, trading_stats: Dict) -> None:
        """Построение графиков результатов"""

        if not data or not self.trades:
            print("Недостаточно данных для построения графиков")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Результаты Parabolic SAR стратегии', fontsize=16)

            # График кривой эквити
            if self.equity_curve:
                timestamps = [point['timestamp'] for point in self.equity_curve]
                capitals = [point['capital'] for point in self.equity_curve]

                axes[0, 0].plot(timestamps, capitals, label='Капитал', linewidth=2, color='blue')
                axes[0, 0].axhline(y=self.config['initial_capital'], color='r',
                                 linestyle='--', alpha=0.5, label='Начальный капитал')
                axes[0, 0].set_title('Кривая эквити')
                axes[0, 0].set_ylabel('Капитал ($)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # График просадки
            if self.equity_curve:
                drawdowns = [point['drawdown'] for point in self.equity_curve]
                axes[0, 1].fill_between(timestamps, drawdowns, alpha=0.3, color='red')
                axes[0, 1].plot(timestamps, drawdowns, color='red', linewidth=1)
                axes[0, 1].set_title('Просадка')
                axes[0, 1].set_ylabel('Просадка (%)')
                axes[0, 1].grid(True, alpha=0.3)

            # Распределение P&L сделок
            pnl_values = [trade.pnl for trade in self.trades]
            axes[1, 0].hist(pnl_values, bins=20, alpha=0.7, edgecolor='black', color='green')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Распределение P&L сделок')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Количество сделок')
            axes[1, 0].grid(True, alpha=0.3)

            # Статистика по причинам выхода
            if trading_stats and 'exit_stats' in trading_stats:
                exit_reasons = list(trading_stats['exit_stats'].keys())
                exit_counts = [trading_stats['exit_stats'][reason]['count'] for reason in exit_reasons]

                colors = ['lightblue', 'lightcoral', 'lightgreen']
                axes[1, 1].pie(exit_counts, labels=[r.replace('_', ' ').title() for r in exit_reasons],
                              autopct='%1.1f%%', colors=colors)
                axes[1, 1].set_title('Причины выхода из сделок')

            plt.tight_layout()

            if OUTPUT_CONFIG['save_results']:
                plot_path = os.path.join(OUTPUT_CONFIG['results_dir'], 'sar_backtest_charts.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Графики сохранены в: {plot_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Ошибка при построении графиков: {e}")

def main():
    """Основная функция"""

    try:
        # Создание движка бэктестинга
        backtest = SARBacktestEngine(RISK_CONFIG)

        # Запуск бэктестинга
        results = backtest.run_backtest()

        if results:
            print("\nSAR бэктестинг завершен успешно!")
        else:
            print("\nОшибка при выполнении бэктестинга")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\nКритическая ошибка: {e}")

if __name__ == "__main__":
    main()
