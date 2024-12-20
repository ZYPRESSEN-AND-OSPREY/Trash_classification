import cv2
import torch
import serial
from ultralytics import YOLO
import numpy as np
import threading
import time
import subprocess
import sys

# 全局控制变量
DEBUG_WINDOW = False
ENABLE_SERIAL = True
CONF_THRESHOLD = 0.9  # 置信度阈值

# 串口配置
STM32_PORT = '/dev/ttyS0'  # STM32串口(TX-> GPIO14,RX->GPIO15)
STM32_BAUD = 115200

# 串口通信协议
FRAME_HEADER = b''  # 帧头(STM32)可为空
FRAME_FOOTER = b''  # 帧尾(STM32)可为空

def setup_gpu():
    if not torch.cuda.is_available():
        return False, "未检测到GPU，将使用CPU进行推理"
    
    device_name = torch.cuda.get_device_name(0)
    return True, f"已启用GPU: {device_name}"

class SerialManager:
    def __init__(self):
        self.stm32_port = None
        self.is_running = True
        self.last_stm32_send_time = 0
        self.MIN_SEND_INTERVAL = 0.1  # 最小发送间隔（秒）
        
        # 垃圾计数和记录相关
        self.garbage_count = 0  # 垃圾计数器
        self.detected_items = []  # 存储检测到的垃圾记录
        
        # 防重复计数和稳定性检测相关
        self.last_count_time = 0  # 上次计数的时间
        self.COUNT_COOLDOWN = 5.0  # 计数冷却时间（秒）
        self.is_counting_locked = False  # 计数锁定状态
        self.last_detected_type = None  # 上次检测到的垃圾类型
        
        # 稳定性检测相关(当前物体)
        self.current_detection = None  # 当前正在检测的物体类型
        self.detection_start_time = 0  # 开始检测的时间
        self.STABILITY_THRESHOLD = 1.0  # 稳定识别所需时间（秒）
        self.stable_detection = False  # 是否已经稳定识别
        self.detection_lost_time = 0  # 丢失检测的时间
        self.DETECTION_RESET_TIME = 0.5  # 检测重置时间（秒）

        # 初始化STM32串口
        if ENABLE_SERIAL:
            try:
                self.stm32_port = serial.Serial(
                    STM32_PORT, 
                    STM32_BAUD, 
                    timeout=0.1,
                    write_timeout=0.1
                )
                print(f"STM32串口已初始化: {STM32_PORT}")
            except Exception as e:
                print(f"STM32串口初始化失败: {str(e)}")
                self.stm32_port = None

        # 启动数据接收线程
        if self.stm32_port:
            self.receive_thread = threading.Thread(target=self.receive_stm32_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()

    def receive_stm32_data(self):
        """接收STM32数据的线程函数"""
        buffer_size = 10240  # 设置合理的缓冲区大小
        
        while self.is_running and self.stm32_port and self.stm32_port.is_open:
            try:
                # 检查串口是否有数据可读
                if self.stm32_port.in_waiting > 0:
                    # 读取数据，限制读取大小
                    data = self.stm32_port.read(min(self.stm32_port.in_waiting, buffer_size))
                    
                    if data:
                        try:
                            # 尝试解码数据（去除无效字符）
                            decoded_data = data.decode('utf-8', errors='replace').strip()
                            # 过滤掉全是 null 字符或 0xFF 的数据
                            if any(c not in ['\x00', '\xff'] for c in decoded_data):
                                # 将数据转换为十六进制字符串
                                hex_data = ' '.join(f'0x{byte:02X}' for byte in data)
                                print(f"接收到的原始数据: {hex_data}")
                                print(f"解码后的数据: {decoded_data}")
                        except UnicodeDecodeError as e:
                            hex_data = ' '.join(f'0x{byte:02X}' for byte in data)
                            print(f"数据解码错误: {str(e)}")
                            print(f"原始数据: {hex_data}")
                    
                    # 清理缓冲区
                    self.stm32_port.reset_input_buffer()
                time.sleep(0.01)
                
            except serial.SerialException as e:
                print(f"串口通信错误: {str(e)}")
                # 尝试重新打开串口
                try:
                    if self.stm32_port.is_open:
                        self.stm32_port.close()
                    time.sleep(1)  # 等待一秒后重试
                    self.stm32_port.open()
                    print("串口重新打开成功")
                except Exception as reopen_error:
                    print(f"串口重新打开失败: {str(reopen_error)}")
                    break  # 如果重新打开失败，退出循环
                    
            except Exception as e:
                print(f"其他错误: {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                    
        print("串口接收线程终止")

    def check_detection_stability(self, garbage_type):
        """检查检测的稳定性"""
        current_time = time.time()
        
        # 如果检测到了新的物体类型，或者检测中断超过重置时间
        if (garbage_type != self.current_detection or 
            (current_time - self.detection_lost_time > self.DETECTION_RESET_TIME and 
             self.detection_lost_time > 0)):
            # 重置检测状态
            self.current_detection = garbage_type
            self.detection_start_time = current_time
            self.stable_detection = False
            self.detection_lost_time = 0
            return False
        
        # 如果已经达到稳定识别时间
        if (current_time - self.detection_start_time >= self.STABILITY_THRESHOLD and 
            not self.stable_detection):
            self.stable_detection = True
            return True
            
        return self.stable_detection

    def can_count_new_garbage(self, garbage_type):
        """检查是否可以计数新垃圾"""
        current_time = time.time()
        
        # 检查稳定性
        if not self.check_detection_stability(garbage_type):
            return False
        
        # 如果是新的垃圾类型，重置锁定状态
        if garbage_type != self.last_detected_type:
            self.is_counting_locked = False
            self.last_detected_type = garbage_type
        
        # 检查是否在冷却时间内
        if self.is_counting_locked:
            if current_time - self.last_count_time >= self.COUNT_COOLDOWN:
                self.is_counting_locked = False  # 解除锁定
            else:
                return False
        
        return True

    def update_garbage_count(self, garbage_type):
        """更新垃圾计数"""
        if not self.can_count_new_garbage(garbage_type):
            return
        
        self.garbage_count += 1
        self.detected_items.append({
            'count': self.garbage_count,
            'type': garbage_type,
            'quantity': 1,
            'status': "正确"
        })
        
        # 更新计数相关的状态
        self.last_count_time = time.time()
        self.is_counting_locked = True
        self.last_detected_type = garbage_type

    def send_to_stm32(self, class_id):
        """发送数据到STM32"""
        if not self.stm32_port or not self.stm32_port.is_open:
            return
    
        current_time = time.time()
        if current_time - self.last_stm32_send_time < self.MIN_SEND_INTERVAL:
            return
    
        try:
            self.stm32_port.reset_input_buffer()
            self.stm32_port.reset_output_buffer()
            
            # 将class_id转换为单字节十六进制
            hex_data = bytes([class_id])
            data = hex_data
            
            # 发送数据
            self.stm32_port.write(data)
            self.stm32_port.flush()
            
            self.last_stm32_send_time = current_time
            print(f"发送数据: {' '.join([f'0x{b:02X}' for b in data])}")
            print(f"发送的分类ID (十六进制): 0x{class_id:02X}")
            print(f"发送的分类ID (十进制): {class_id}")
            
        except Exception as e:
            print(f"串口发送错误: {str(e)}")

    def cleanup(self):
        """清理串口资源"""
        self.is_running = False
        if self.stm32_port and self.stm32_port.is_open:
            self.stm32_port.close()
            
class WasteClassifier:
    def __init__(self):
        # 分类名称
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }
        
        # 细分类到大分类的映射 - 移除,因为我们直接使用四大类
        self.category_mapping = None
        
        # 分类名称对应的描述(可选)
        self.category_descriptions = {
            0: "厨余垃圾",
            1: "可回收利用垃圾",
            2: "有害垃圾",
            3: "其他垃圾"
        }

    def get_category_info(self, class_id):
        """
        获取给定类别ID的分类信息
        返回: (分类名称, 分类描述)
        """
        category_name = self.class_names.get(class_id, "未知分类")
        description = self.category_descriptions.get(class_id, "未知描述")
        
        return category_name, description

    def print_classification(self, class_id):
        """打印分类信息"""
        category_name, description = self.get_category_info(class_id)
        print(f"\n垃圾分类信息:")
        print(f"分类类别: {category_name}")
        print(f"分类说明: {description}")
        print("-" * 30)
        
        return f"{category_name}"
class YOLODetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

        # 更新为四大类
        self.class_names = {
            0: '厨余垃圾',
            1: '可回收垃圾',
            2: '有害垃圾',
            3: '其他垃圾'
        }

        # 为每个类别指定固定的颜色
        self.colors = {
            0: (86, 180, 233),    # 厨余垃圾 - 蓝色
            1: (230, 159, 0),     # 可回收垃圾 - 橙色
            2: (240, 39, 32),     # 有害垃圾 - 红色
            3: (0, 158, 115)      # 其他垃圾 - 绿色
        }
        self.serial_manager = SerialManager()

    def detect(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD)
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) > 0:
                confidences = [box.conf[0].item() for box in boxes]
                max_conf_idx = np.argmax(confidences)
                box = boxes[max_conf_idx]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                waste_classifier = WasteClassifier()
                category_id, description = waste_classifier.get_category_info(class_id)
                display_text = f"{category_id}({description})"
                
                color = self.colors.get(class_id, (255, 255, 255))
                
                if DEBUG_WINDOW:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    label = f"{display_text} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                print(f"检测到物体:")
                print(f"置信度: {confidence:.2%}")
                print(f"边界框位置: ({x1}, {y1}), ({x2}, {y2})")
                print(f"中心点位置: ({center_x}, {center_y})")
                print("-" * 30)
                self.serial_manager.send_to_stm32(class_id)
                self.serial_manager.update_garbage_count(display_text)
        
        return frame
def find_camera():
    """查找可用的摄像头"""
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"成功找到可用摄像头，索引为: {index}")
            return cap
        cap.release()
    
    print("错误: 未找到任何可用的摄像头")
    return None

def main():
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)
    
    detector = YOLODetector(
        model_path='best.pt'
    )
    
    cap = find_camera()
    if not cap:
        return
    
    if DEBUG_WINDOW:
        window_name = 'YOLOv8检测'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print(f"- 调试窗口: {'开启' if DEBUG_WINDOW else '关闭'}")
    print(f"- 串口输出: {'开启' if ENABLE_SERIAL else '关闭'}")
    print("- 按 'q' 键退出程序")
    print("-" * 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            frame = detector.detect(frame)
            
            if DEBUG_WINDOW:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n程序正常退出")
                    break
            
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        # 清理资源
        if hasattr(detector, 'serial_manager'):
            detector.serial_manager.cleanup()
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
