#最简化数据加载

import pandas as pd
from src.exceptions import DataLoadError  # 用自定义异常来报错

def load_data(path):
    try:
        df = pd.read_csv(path)  # 尝试读取 CSV 文件
        print(f"Loaded {len(df)} sampels") # 简单打印一下加载了多少条
        return df
    except Exception as e:
        #任何一场都包装成DataLoadError 抛出
        raise DataLoadError(f"Failed to load {path}:{e}")