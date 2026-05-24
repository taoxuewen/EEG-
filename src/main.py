"""
EEG 运动想象分析系统 — 主入口

使用 CSP (Common Spatial Patterns) 算法对运动想象 EEG 数据进行
特征提取和分类，提供图形化界面进行动态 EEG 信号浏览和结果分析。
"""

import tkinter as tk
import sys
import os

# Ensure src/ is on path for imports
sys.path.insert(0, os.path.dirname(__file__))

from gui import EEGViewerApp


def main():
    root = tk.Tk()
    app = EEGViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
