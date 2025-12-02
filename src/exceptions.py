# 自定义两个异常类型，用来在工程里更清晰地表示错误类型。


class DataLoadError(Exception):
    """数据加载相关的错误"""
    pass


class ModelNotFittedError(Exception):
    """模型还没训练好就拿来预测时抛的错误"""
    pass
