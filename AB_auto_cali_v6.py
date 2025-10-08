import re
import time
import cv2
import keyboard
import numpy as np
import pyautogui
import random
from paddleocr import PaddleOCR
from PIL import Image
import json
"""
    本程序用于自动校炮，初次使用需要设置以下内容
    1. 超参数 距离数字区域截图坐标
    2. load.json中请添加需要使用的规则
    规则格式示例：
    {
        "name": "120L55_dm63", #名字随便，切换时候在终端看的。
        "path": "./Ballistic/it_leopard_2a7_hungary/dm53.txt" #路径对应就行。
    }

    3. 标尺规则已经接入数据库。
    纯ai视觉+模拟输入，无内存读取。
"""

# Start 超参数
# 1. 读取 JSON 文件
with open('config.json', 'r', encoding='utf-8') as f:  # 假设你的文件名是 config.json
    config = json.load(f)

# 2. 从 JSON 提取各个变量
x1, y1 = config["x1y1"]
x2, y2 = config["x2y2"]
max_det_num = config["max_det_num"]
min_cali_distance = config["min_cali_distance"]
alpha = config["alpha"] * 0.02
safe_mode = config["safe_mode"] == "True"  # 将字符串转为布尔值
debug_mode = config["debug_mode"] == "True"
load_path = config["load_path"]
# End 超参数




def load_first_columns(json_path):
    """
    从指定的JSON中读取所有文件路径，提取每个文件的第一列（忽略第一行），并以name为键保存。
    返回一个字典： { name: [第一列数据列表], ... }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)

    data_dict = {}

    for item in configs:
        name = item['name']
        txt_path = item['path']

        first_col = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as ftxt:
                # 先跳过第一行
                next(ftxt)
                for line in ftxt:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    try:
                        first_col.append(float(parts[0]))
                    except (IndexError, ValueError):
                        continue
            data_dict[name] = first_col
        except FileNotFoundError:
            print(f"⚠️ 文件未找到: {txt_path}")
        except Exception as e:
            print(f"⚠️ 读取 {txt_path} 出错: {e}")

    return data_dict


def number_is_right(number, rule):
    """
    判断数字是否在规则范围内
    """
    if min_cali_distance < number < rule[-1]:
        return True
    else:
        return False

def get_gear_from_number(number, rule):
    """
    根据输入的数字判断对应的挡位（支持浮点数档位）
    
    参数:
        number: 输入的数字
        rule: 规则对象，包含档位字典
    
    返回:
        对应的挡位数字（浮点数），如果无法判断则返回None
    """
    
    # 获取所有档位，按档位号排序
    rule = {i+1: v for i, v in enumerate(rule)}
    gears = sorted(rule.keys())
    
    # 如果输入值小于最小档位值，与0和最小档位进行线性插值
    if number <= rule[gears[0]]:
        min_gear = gears[0]
        min_value = rule[min_gear]
        # 在0档位（值为0）和最小档位之间进行线性插值
        ratio = number / min_value
        precise_gear = ratio * min_gear
        return precise_gear
    
    # 如果输入值大于最大档位值，返回最大档位
    if number >= rule[gears[-1]]:
        return float(gears[-1])
    
    # 在档位之间进行线性插值
    for i in range(len(gears) - 1):
        lower_gear = gears[i]
        upper_gear = gears[i + 1]
        lower_value = rule[lower_gear]
        upper_value = rule[upper_gear]
        
        # 如果输入值在当前两个档位之间
        if lower_value <= number <= upper_value:
            # 线性插值计算精确档位
            ratio = (number - lower_value) / (upper_value - lower_value)
            precise_gear = lower_gear + ratio
            # precise_gear = round(precise_gear, 3)
            return precise_gear
    
    # 如果没有找到合适的区间，返回0
    return 0

class FastScreenOCR:
    def __init__(self):
        # 初始化PaddleOCR，只使用英文模型，关闭角度分类器以提高速度
        self.ocr = PaddleOCR(use_textline_orientation=False, lang="en", show_log=False, det_model_dir='.model/det/en/en_PP-OCRv3_det_infer',
    rec_model_dir='.model/rec/en/en_PP-OCRv3_rec_infer', cls_model_dir='.model/cls/ch_ppocr_mobile_v2.0_cls_infer')
        # 禁用pyautogui的安全检查以提高速度
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

    def capture_region(self, x1, y1, x2, y2):
        """
        快速截取指定区域
        参数:
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
        返回:
            PIL Image对象
        """
        # 计算宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 使用pyautogui快速截图
        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        return screenshot

    def preprocess_image(self, image):
        """
        图像预处理以提取红色元素
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 确保是RGB格式
        if len(img_array.shape) == 3:
            # 转换为HSV色彩空间，更适合颜色检测
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            # 如果是灰度图，无法提取红色，返回原图
            return img_array
        
        # 定义红色的HSV范围
        # 红色在HSV中有两个范围（因为红色在色相环的两端）
        # 范围1：较低的红色值 [H色相, S饱和度, V明度]
        # 红色在 HSV 空间的两段范围
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([150, 50, 70])
        upper_red2 = np.array([180, 255, 255])

        """
        lower_red1 = np.array([h1_min, s1_min, v1_min])
        upper_red1 = np.array([h1_max, s1_max, v1_max])
        lower_red2 = np.array([h2_min, s2_min, v2_min])
        upper_red2 = np.array([h2_max, s2_max, v2_max])
        """
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # 合并两个掩码
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        return red_mask

    def extract_numbers(self, image):
        """
        从图像中提取数字
        """
        try:
            # 转换PIL图像为numpy数组（PaddleOCR需要numpy数组）
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            # 使用PaddleOCR进行识别
            result = self.ocr.ocr(img_array, cls=False)

            # 提取所有识别到的文本
            all_text = ""
            if result and result[0]:
                for line in result[0]:
                    if line[1][1] > 0.5:  # 置信度阈值
                        all_text += line[1][0] + " "

            print(f"识别到的文本: {all_text}")

            # 使用正则表达式提取数字（包括小数和负数）
            # 提取带km单位的数字（只要纯数字部分）
            km_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*k[\W_]*m?", all_text)

            # 换算为m单位的数字且整数
            numbers = int(float(km_numbers[0]) * 1000)

            # 转换为浮点数列表，取绝对值，然后返回最大值
            if type(numbers) == int:
                return numbers
            elif len(numbers) > 1:
                abs_numbers = [abs(num) for num in numbers]
                return max(abs_numbers)
            else:
                return 0  # 如果没有检测到数字，返回0


        except Exception as e:
            print(f"OCR识别错误: {e}")
            return 0

    def screenshot_and_ocr(self, x1, y1, x2, y2, preprocess=True):
        """
        完整的截图和OCR流程
        参数:
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
            preprocess: 是否进行图像预处理
        返回:
            识别到的数字列表
        """
        

        # 截图
        screenshot = self.capture_region(x1, y1, x2, y2)

        # 图像预处理（可选）
        if preprocess:
            processed_image = self.preprocess_image(screenshot)
        else:
            processed_image = np.array(screenshot)

        # OCR识别
        number = self.extract_numbers(processed_image)

        return number

    def save_debug_image(self, x1, y1, x2, y2, filename="debug.png"):
        """
        保存调试图像，用于检查截图效果
        """
        screenshot = self.capture_region(x1, y1, x2, y2)
        processed = self.preprocess_image(screenshot)

        # 保存原图和处理后的图
        screenshot.save(f"original_{filename}")
        cv2.imwrite(f"processed_{filename}", processed)
        print(f"调试图像已保存: original_{filename}, processed_{filename}")



if __name__ == "__main__":
    rules_dic = load_first_columns(load_path)
    print("已读取的项目:", list(rules_dic.keys()))
    
    # 定义规则列表和名称映射
    rule_choices = list(rules_dic.keys())
    
    # 初始化当前规则索引
    current_rule_index = 0
    current_rule_name = list(rules_dic.keys())[0]
    rule = rules_dic[current_rule_name]
    
    print(f"使用 {current_rule_name} 火炮规则")
    
    # 创建OCR实例
    ocr = FastScreenOCR()

    print("程序已启动，实时监控键盘按键...")
    print("按下 'z' 键执行自动校炮")
    print("按下 'delete' 键切换火炮规则")
    print("按下 'ctrl+c' 或关闭窗口退出程序")

    def on_z_pressed():
        """当按下z键时执行的函数"""
        # print("\n检测到z键按下，开始截图和OCR识别...")
        start_time = time.time()

        if safe_mode:   # 安全模式，保证精度
            last_num = None

            for i in range(max_det_num):
                # 执行截图和OCR，获取当前数字，失败为0
                number = ocr.screenshot_and_ocr(x1, y1, x2, y2)

                if number_is_right(number, rule):   # 如果数字合理，则进入判断
                    if number == last_num:  # 如果上一个数字和当前数字相同
                        break
                    else:
                        last_num = number   # 不相同，更新last_num
        else:
            for i in range(max_det_num):
                # 执行截图和OCR，获取当前数字，失败为0
                number = ocr.screenshot_and_ocr(x1, y1, x2, y2)
                if number_is_right(number, rule):
                    break


        if number_is_right(number, rule):
            gear = get_gear_from_number(number, rule)
            print(f"识别到的 距离|挡位: {number}|{gear}")
            pyautogui.scroll(int((120 * len(rule) + random.randint(0, 240)) * alpha))  # 反作弊
            time.sleep(0.08) # 间隔，可适当修改
            pyautogui.scroll(int(-120 * gear * alpha))

        else:
            print(f"不符合规则的距离：{number}")
        
        
        end_time = time.time()
        print(f"处理时间: {end_time - start_time:.3f}秒")

        # 如果需要调试，可以保存图像
        if debug_mode:
            print('[Debug]:图像已保存至当前目录')
            ocr.save_debug_image(x1, y1, x2, y2)

    def on_del_pressed():
        """当按下delete键时切换规则"""
        global current_rule_index, current_rule_name, rule
        
        # 切换到下一个规则
        current_rule_index = (current_rule_index + 1) % len(rule_choices)
        current_rule_name = rule_choices[current_rule_index]
        rule = rules_dic[current_rule_name]
        
        print(f"\n已切换到: {current_rule_name} 火炮规则")
        # print(rule)

    # 注册键盘事件监听
    keyboard.add_hotkey("z", on_z_pressed)
    keyboard.add_hotkey("delete", on_del_pressed)

    try:
        # 保持程序运行，监听键盘事件
        print("程序运行中，按 Ctrl+C 退出...")
        keyboard.wait()  # 无限等待，直到程序被中断
    except KeyboardInterrupt:
        print("\n程序被中断退出...")
    finally:
        print("程序已结束")
