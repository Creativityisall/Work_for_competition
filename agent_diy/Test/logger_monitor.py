import logging
import matplotlib.pyplot as plt

import logging

def init_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():          # 防止重复初始化
        return logger

    logger.setLevel(logging.DEBUG)
    log_file = f"./Test/backup/test.log"

    # 新建 FileHandler，mode='w' 表示覆盖写
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_fmt = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s"
    file_handler.setFormatter(logging.Formatter(file_fmt))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    return logger

class Monitor:
    def __init__(self):
        self.data = {}

    def put_data(self, data):
        """将数据存入监控器"""
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        
    def get_data(self):
        """获取当前监控数据"""
        return self.data
    
    def clear(self):
        """清空监控数据"""
        self.data.clear()

    def draw(self):
        """绘制监控数据"""
        if not self.data:
            print("No data to draw.")
            return

        for label, value in self.data.items():
            if isinstance(value, tuple) and len(value) == 2:
                x, y = value
            else:
                x, y = None, value   # 自动 x
            plt.plot(x, y, label=str(label), marker='o', markersize=3)

        plt.title("Monitor Data")
        plt.xlabel("Index / X")
        plt.ylabel("Value / Y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def init_monitor():
    return Monitor()