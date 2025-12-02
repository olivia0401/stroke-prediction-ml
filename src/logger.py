"""
Centralized logging system for stroke prediction pipeline.
Replaces print() statements with structured logging.
"""

import logging # python自带日志模块
from pathlib import Path #处理路径的工具

def setup_logger(name: str):
    # 确保 logs 目录存在（如果不存在就创建，存在就啥也不干）
    Path("logs").mkdir(exist_ok=True)

    # 用 basicConfig 一次性配置好 logging 的“根 logger”
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()            
        ],
    )

    # 返回一个带这个 name 的 logger 对象，以后用 logger.info(...) 之类 
    return logging.getLogger(name)


    