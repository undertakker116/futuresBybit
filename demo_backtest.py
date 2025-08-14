#!/usr/bin/env python3
"""
Демо версия SAR бэктестинга с MACD фильтрацией (локальные данные)
"""

# Импортируем все нужные модули
from backtest_system import SARBacktestEngine
from demo_data_fetcher import DemoDataFetcher
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_demo_backtest():
    """Запуск демо бэктестинга с локальными данными"""
    
    try:
        logger.info("=== ДЕМО БЭКТЕСТИНГ SAR + MACD ===")
        
        # Создаем демо версию бэктестера
        config = {
            'initial_capital': 10000,
            'risk_per_trade': 5.0,
            'stop_loss_atr_multiplier_long': 2.5,
            'stop_loss_atr_multiplier_short': 2.5,
            'take_profit_atr_multiplier_long': 5.0,
            'take_profit_atr_multiplier_short': 5.0,
            'symbols': ["FARTCOINUSDT", "SOLUSDT"],
            'timeframe': '15m',
            'start_date': '2025-05-01',
            'end_date': '2025-08-14'
        }
        
        # Заменяем data_fetcher на демо версию
        backtester = SARBacktestEngine(config)
        backtester.data_fetcher = DemoDataFetcher()
        
        # Запускаем бэктестинг
        results = backtester.run_backtest()
        
        if results:
            logger.info("=== РЕЗУЛЬТАТЫ ДЕМО БЭКТЕСТИНГА ===")
            logger.info("Демо бэктестинг завершен успешно!")
        else:
            logger.error("Демо бэктестинг завершился с ошибкой")
            
    except Exception as e:
        logger.error(f"Ошибка при выполнении демо бэктестинга: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo_backtest()