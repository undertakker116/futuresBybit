#!/usr/bin/env python3
"""
ГЛАВНЫЙ ФАЙЛ ЗАПУСКА SAR БЭКТЕСТИНГ СИСТЕМЫ
Автоматически генерирует данные и запускает бэктестинг с MACD фильтрацией
"""

import os
import sys
import logging
from typing import Dict, Any
import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Красивый баннер приветствия"""
    print("\n" + "="*80)
    print("🚀 SAR БЭКТЕСТИНГ СИСТЕМА С MACD ФИЛЬТРАЦИЕЙ")
    print("="*80)
    print(f"📅 Период: {config.START_DATE} до {config.END_DATE}")
    print(f"💎 Символы: {', '.join(config.SYMBOLS)}")
    print(f"⏰ Таймфрейм: {config.PRIMARY_TIMEFRAME}")
    print(f"📊 Всего символов: {len(config.SYMBOLS)}")
    print("="*80)

def check_dependencies():
    """Проверка зависимостей"""
    try:
        import pandas
        import numpy
        import pybit
        import matplotlib
        import seaborn
        logger.info("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        logger.error(f"❌ Отсутствует зависимость: {e}")
        logger.info("Установите зависимости: pip install pandas numpy pybit matplotlib seaborn")
        return False

def generate_demo_data():
    """Генерация демо данных"""
    try:
        logger.info("📊 Генерация демо данных...")
        
        # Импортируем и запускаем генератор
        from demo_data_generator import save_demo_data
        save_demo_data()
        
        logger.info("✅ Демо данные успешно сгенерированы")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации данных: {e}")
        return False

def run_backtest_analysis():
    """Запуск бэктестинга"""
    try:
        logger.info("🎯 Запуск SAR бэктестинга с MACD фильтрацией...")
        
        # Импортируем и запускаем бэктестер
        from demo_backtest import run_demo_backtest
        run_demo_backtest()
        
        logger.info("✅ Бэктестинг завершен успешно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка бэктестинга: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_results():
    """Отображение информации о результатах"""
    print("\n" + "="*80)
    print("📈 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*80)
    
    # Проверяем наличие файлов результатов
    results_files = [
        "backtest_results/sar_trades.json",
        "backtest_results/sar_backtest_charts.png",
        "backtest.log"
    ]
    
    print("📁 Созданные файлы:")
    for file_path in results_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (не найден)")
    
    print("\n📊 Демо данные сохранены в папке: demo_data/")
    print("📈 Графики и анализ: backtest_results/")
    print("📝 Лог выполнения: backtest.log")
    
    print("\n" + "="*80)
    print("🎉 АНАЛИЗ ЗАВЕРШЕН! Проверьте файлы результатов.")
    print("="*80 + "\n")

def main():
    """Главная функция запуска"""
    print_banner()
    
    # Проверяем зависимости
    if not check_dependencies():
        return False
    
    try:
        # Шаг 1: Генерация данных
        if not generate_demo_data():
            logger.error("Не удалось сгенерировать данные")
            return False
        
        # Шаг 2: Запуск бэктестинга
        if not run_backtest_analysis():
            logger.error("Не удалось выполнить бэктестинг")
            return False
        
        # Шаг 3: Показ результатов
        show_results()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Выполнение прервано пользователем")
        return False
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """Точка входа в программу"""
    success = main()
    sys.exit(0 if success else 1)