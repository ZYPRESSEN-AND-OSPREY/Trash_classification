import cv2
import json
import tensorflow as tf
import numpy as np

# 配置参数
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# 垃圾分类颜色映射
CATEGORY_COLORS = {
    "其他垃圾": (128, 128, 128),  # 灰色
    "厨余垃圾": (0, 255, 0),      # 绿色
    "可回收物": (0, 0, 255),      # 蓝色
    "有害垃圾": (0, 0, 128)       # 红色
}

def load_labels(label_file):
    """加载标签映射"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    return labels_dict

def get_category_color(class_name):
    """获取垃圾分类类别的颜色"""
    for category, color in CATEGORY_COLORS.items():
        if category in class_name:
            return color
    return (0, 255, 0)  # 默认绿色

class GarbageDetector:
    def __init__(self, model_path, label_file):
        # 加载模型
        self.model = tf.keras.models.load_model(model_path)
        
        # 加载标签
        self.labels = load_labels(label_file)
        
    def preprocess_image(self, image):
        """预处理图像"""
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect(self, image):
        """执行检测"""
        orig_h, orig_w = image.shape[:2]
        processed_image = self.preprocess_image(image)
        box_pred, class_pred = self.model.predict(processed_image, verbose=0)
        
        # 获取前三个最可能的类别
        top_3_indices = np.argsort(class_pred[0])[-3:][::-1]
        top_3_confidences = class_pred[0][top_3_indices]
        top_3_classes = [self.labels[str(idx)] for idx in top_3_indices]
        
        # 主要预测结果
        class_id = top_3_indices[0]
        confidence = top_3_confidences[0]
        
        if confidence > CONFIDENCE_THRESHOLD:
            # 获取边界框预测
            x1, y1, x2, y2 = box_pred[0]
            
            # 转换边界框坐标到原始图像尺寸
            x1 = int(x1 * orig_w)
            y1 = int(y1 * orig_h)
            x2 = int(x2 * orig_w)
            y2 = int(y2 * orig_h)
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            return {
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'class_id': int(class_id),
                'class_name': self.labels[str(class_id)],
                'confidence': float(confidence),
                'top_3_predictions': list(zip(top_3_classes, top_3_confidences))
            }
        return None

def draw_results(image, result):
    """在图像上绘制检测结果"""
    if result:
        # 获取主要类别的颜色
        color = get_category_color(result['class_name'])
        
        # 绘制边界框
        x1, y1, x2, y2 = result['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制中心点
        center_x, center_y = result['center']
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 计算文本框的位置和大小
        padding = 10
        line_height = 30
        
        # 绘制主要预测结果
        main_label = f"{result['class_name']} ({result['confidence']:.2f})"
        cv2.putText(image, main_label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 绘制中心点坐标
        coord_text = f"Center: ({center_x}, {center_y})"
        cv2.putText(image, coord_text, (x1, y2+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 在右侧显示详细信息
        info_x = image.shape[1] - 400
        info_y = 30
        
        # 添加标题
        cv2.putText(image, "Detection Results:", (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 显示前三个预测结果
        for i, (class_name, conf) in enumerate(result['top_3_predictions']):
            text_color = get_category_color(class_name)
            prediction_text = f"{i+1}. {class_name}: {conf:.2f}"
            cv2.putText(image, prediction_text,
                       (info_x, info_y + (i+1)*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # 打印结果到控制台
        print("\n检测结果:")
        print(f"位置: 中心坐标 ({center_x}, {center_y})")
        print("\n置信度最高的三个类别:")
        for i, (class_name, conf) in enumerate(result['top_3_predictions']):
            category = class_name.split('/')[0]
            specific = class_name.split('/')[1]
            print(f"{i+1}. {category:<10} - {specific:<15} : {conf:.2f}")
    
    return image

def process_camera():
    """处理摄像头输入"""
    cap = cv2.VideoCapture(1)
    
    detector = GarbageDetector(
        model_path='garbage_detector.h5',
        label_file='garbage_classify_rule.json'
    )
    
    print("\n=== 垃圾分类实时检测系统 ===")
    print("按 'q' 退出程序")
    print("检测结果将实时显示在画面和控制台中")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            result = detector.detect(frame)
            frame = draw_results(frame, result)
            
            # 显示操作提示
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Garbage Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n程序已退出")
                break
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_image(image_path):
    """处理单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    detector = GarbageDetector(
        model_path='garbage_detector.h5',
        label_file='garbage_classify_rule.json'
    )
    
    result = detector.detect(image)
    image = draw_results(image, result)
    
    cv2.imshow('Garbage Detection', image)
    print("\n按任意键退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n处理图片: {image_path}")
        process_image(image_path)
    else:
        process_camera()

if __name__ == '__main__':
    main()
