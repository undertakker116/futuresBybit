#!/usr/bin/env python3
"""
Демо версия SAR бэктестинга с MACD фильтрацией (локальные данные)
"""

# Импортируем все нужные модули
from backtest_system import SARBacktestEngine
from demo_data_fetcher import DemoDataFetcher
import config
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
        config_dict = {
            'initial_capital': config.RISK_CONFIG['initial_capital'],
            'risk_per_trade': config.RISK_CONFIG['risk_per_trade'],
            'stop_loss_atr_multiplier_long': config.RISK_CONFIG['stop_loss_atr_multiplier_long'],
            'stop_loss_atr_multiplier_short': config.RISK_CONFIG['stop_loss_atr_multiplier_short'],
            'take_profit_atr_multiplier_long': config.RISK_CONFIG['take_profit_atr_multiplier_long'],
            'take_profit_atr_multiplier_short': config.RISK_CONFIG['take_profit_atr_multiplier_short'],
            'symbols': config.SYMBOLS,
            'timeframe': config.PRIMARY_TIMEFRAME,
            'start_date': config.START_DATE,
            'end_date': config.END_DATE
        }
        
        # Заменяем data_fetcher на демо версию
        backtester = SARBacktestEngine(config_dict)
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