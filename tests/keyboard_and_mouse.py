# import moudles 导入pywin32的 Dispatch 函数
import time
from win32com.client import Dispatch
import pyautogui


if __name__ == "__main__":
    time.sleep(5)
    pyautogui.FAILSAFE = False
    # 移动鼠标到屏幕坐标(100, 100)
    pyautogui.moveTo(200, 200)

    # 按下键盘按键'a'
    pyautogui.press("w")
