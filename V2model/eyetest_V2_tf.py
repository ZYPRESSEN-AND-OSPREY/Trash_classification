import cv2
import json
import numpy as np
import tflite_runtime.interpreter as tflite

# 配置参数
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# 垃圾分类颜色映射
CATEGORY_COLORS = {
    "其他垃圾": (128, 128, 128),
    "厨余垃圾": (0, 255, 0),
    "可回收物": (0, 0, 255),
    "有害垃圾": (0, 0, 128)
}

def load_labels(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_category_color(class_name):
    for category, color in CATEGORY_COLORS.items():
        if category in class_name:
            return color
    return (0, 255, 0)

class GarbageDetector:
    def __init__(self, model_path, label_file):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = load_labels(label_file)
    
    def preprocess_image(self, image):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        processed_image = self.preprocess_image(image)
        
        # 设置输入数据
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # 运行推理
        self.interpreter.invoke()
        
        # 获取输出
        box_pred = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_pred = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        
        # 获取前三个最可能的类别
        top_3_indices = np.argsort(class_pred)[-3:][::-1]
        top_3_confidences = class_pred[top_3_indices]
        top_3_classes = [self.labels[str(idx)] for idx in top_3_indices]
        
        # 主要预测结果
        class_id = top_3_indices[0]
        confidence = top_3_confidences[0]
        
        if confidence > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box_pred
            x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
            y1, y2 = int(y1 * orig_h), int(y2 * orig_h)
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
    if result:
        color = get_category_color(result['class_name'])
        x1, y1, x2, y2 = result['bbox']
        center_x, center_y = result['center']
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制中心点
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
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
    cap = cv2.VideoCapture(0)
    detector = GarbageDetector(
        model_path='garbage_detector.tflite',
        label_file='garbage_classify_rule.json'
    )
    
    print("\n=== 垃圾分类实时检测系统 ===")
    print("按 'q' 退出程序")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            result = detector.detect(frame)
            frame = draw_results(frame, result)
            
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

def main():
    process_camera()

if __name__ == '__main__':
    main()
