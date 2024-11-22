import cv2
import tensorflow as tf
import numpy as np
import json

class GarbageDetector:
    def __init__(self, model_path, labels_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        self.categories = {
            '其他垃圾': (128, 128, 128),
            '厨余垃圾': (0, 255, 0),
            '可回收物': (0, 0, 255),
            '有害垃圾': (0, 0, 128)
        }
    
    def preprocess_image(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def get_category(self, label):
        for category in self.categories.keys():
            if label.startswith(category):
                return category
        return None
    
    def detect(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 2
        x1 = center_x - box_size
        y1 = center_y - box_size
        x2 = center_x + box_size
        y2 = center_y + box_size
        
        roi = frame[y1:y2, x1:x2]
        input_data = self.preprocess_image(roi)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        pred_index = np.argmax(output_data[0])
        confidence = float(output_data[0][pred_index])
        label = self.labels[str(pred_index)]
        category = self.get_category(label)
        
        if confidence > 0.5:
            color = self.categories[category]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_category = f"{category}"
            text_label = f"{label.split('/')[-1]}"
            text_conf = f"{confidence:.2f}"
            
            (w1, h1), _ = cv2.getTextSize(text_category, font, font_scale, thickness)
            cv2.putText(frame, text_category, (x1, y1-10), font, font_scale, color, thickness)
            
            (w2, h2), _ = cv2.getTextSize(text_label, font, font_scale, thickness)
            cv2.putText(frame, text_label, (x1, y1-10-h1-5), font, font_scale, color, thickness)
            
            cv2.putText(frame, text_conf, (x1, y1-10-h1-h2-10), font, font_scale, color, thickness)
            
            print("\n检测结果:")
            print(f"类别: {category}")
            print(f"具体物品: {label.split('/')[-1]}")
            print(f"置信度: {confidence:.2%}")
            print("-" * 30)
        
        return frame

def main():
    detector = GarbageDetector(
        model_path='garbage_classifier.tflite',
        labels_path='garbage_classify_rule.json'
    )
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = detector.detect(frame)
        cv2.imshow('Garbage Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
