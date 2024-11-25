import cv2
import numpy as np
import tensorflow as tf
import json
def setup_gpu():
    """
    检测并配置GPU。
    返回:
        bool: 是否使用GPU
        str: 设备信息说明
    """
    try:
        # 检测是否有可用的GPU
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            # 如果没有GPU，则使用CPU
            tf.config.set_visible_devices([], 'GPU')
            return False, "未检测到GPU，将使用CPU进行推理"
            
        # 有GPU的情况下，进行内存配置
        for gpu in gpus:
            try:
                # 允许GPU内存按需增长
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU内存配置失败: {str(e)}")
                return False, "GPU配置失败，将使用CPU进行推理"
        
        # 获取GPU信息
        gpu_info = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = gpu_info.get('device_name', 'Unknown GPU')
        
        return True, f"已启用GPU: {gpu_name}"
        
    except Exception as e:
        print(f"GPU检测过程发生错误: {str(e)}")
        return False, "设备检测失败，将使用CPU进行推理"
    
class GarbageDetectorTFLite:
    def __init__(self, model_path, labels_path):
        # 加载标签映射
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # 加载 TFLite 模型并分配张量
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
        self.interpreter.allocate_tensors()
        
        # 获取输入和输出张量的信息
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 获取模型的输入尺寸
        self.input_shape = self.input_details[0]['shape']
        self.IMG_SIZE = self.input_shape[1]  # 假设输入形状为 [1, 224, 224, 3]
        
        # 定义类别及其对应颜色
        self.categories = {
            '其他垃圾': (128, 128, 128),  # 灰色
            '厨余垃圾': (0, 255, 0),      # 绿色
            '可回收物': (0, 0, 255),      # 红色
            '有害垃圾': (255, 0, 0)       # 蓝色
        }
    
    def preprocess_image(self, img):
        # BGR 转 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        
        # 归一化到 [0,1]
        img = img.astype(np.float32) / 255.0
        
        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)
        return img
    
    def get_category(self, label):
        for category in self.categories.keys():
            if label.startswith(category):
                return category
        return None
    
    def detect(self, frame):
        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 定义中心区域
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x1 = max(center_x - box_size, 0)
        y1 = max(center_y - box_size, 0)
        x2 = min(center_x + box_size, width)
        y2 = min(center_y + box_size, height)
        
        # 提取中心区域并进行预处理
        roi = frame[y1:y2, x1:x2]
        input_data = self.preprocess_image(roi)
        
        # 设置模型输入并运行推理
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # 获取分类输出
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_id = np.argmax(output_data)
        confidence = output_data[class_id]
        
        # 获取标签和类别
        label = self.labels.get(str(class_id), "未知类别")
        category = self.get_category(label)
        
        # 可视化处理
        if confidence > 0.5:  # 置信度阈值
            color = self.categories.get(category, (0, 255, 0))
            
            # 绘制中心检测区域
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            padding = 10
            
            # 准备显示文本
            text_category = f"类别: {category}"
            text_item = f"物品: {label.split('/')[-1]}"
            text_conf = f"置信度: {confidence:.1%}"
            
            # 计算文本位置
            y_offset = y1 - padding
            for text in [text_conf, text_item, text_category]:
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                y_offset -= h + padding
                
                # 添加文本背景
                cv2.rectangle(frame, 
                            (x1, y_offset - padding), 
                            (x1 + w + padding * 2, y_offset + h + padding),
                            (0, 0, 0),
                            -1)
                
                # 绘制文本
                cv2.putText(frame, text, 
                          (x1 + padding, y_offset + h),
                          font, font_scale, color, thickness)
            
            # 打印检测结果
            print("\n检测结果:")
            print(f"类别: {category}")
            print(f"物品: {label.split('/')[-1]}")
            print(f"置信度: {confidence:.1%}")
            print("-" * 30)
        
        return frame

def main():
    use_gpu, device_info = setup_gpu()
    print("\n设备信息:")
    print(device_info)
    print("-" * 30)
    # 初始化检测器
    detector = GarbageDetectorTFLite(
        model_path='garbage_classifier.tflite',
        labels_path='garbage_classify_rule.json'
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    # 设置窗口
    window_name = '垃圾分类检测'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print("\n系统启动:")
    print("- 摄像头已就绪")
    print("- 按 'q' 键退出程序")
    print("- 将物品放置在画面中心区域")
    print("-" * 30)
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 处理帧
            frame = detector.detect(frame)
            
            # 显示结果
            cv2.imshow(window_name, frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n程序正常退出")
                break
                
    except KeyboardInterrupt:
        print("\n检测到键盘中断,程序退出")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
