import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

# 预处理参数
IMG_SIZE = 224  # 保持与原始大小一致
BATCH_SIZE = 32
NUM_CLASSES = 40

# 数据加载器（保持原有格式）
class GarbageDataset:
    def __init__(self, root_dir, txt_file, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        
        # 读取文件列表
        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.data = []
            for line in f:
                img_path, label = line.strip().split()
                img_path = img_path.lstrip('./')
                self.data.append((img_path, int(label)))
                
        print(f"加载了 {len(self.data)} 个样本")
    
    def preprocess_image(self, img_path):
        # 读取和预处理图片
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        
        if self.is_training:
            # 数据增强
            if np.random.random() > 0.5:
                img = tf.image.flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            
        return img
    
    def create_dataset(self):
        def generator():
            while True:  # 添加无限循环
                for img_path, label in self.data:
                    img = self.preprocess_image(img_path)
                    # 使用整图作为目标框
                    box = np.array([0, 0, 1, 1])  # 归一化坐标 [x1, y1, x2, y2]
                    yield img, (box, label)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(4,), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
        )
        
        # 对训练集进行洗牌
        if self.is_training:
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

# 创建检测模型
def create_model():
    # 使用MobileNetV2作为特征提取器
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 添加检测头
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 分类分支
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)
    
    # 检测框分支 (x, y, w, h)
    box_output = layers.Dense(4, activation='sigmoid', name='box_output')(x)
    
    model = models.Model(
        inputs=base_model.input,
        outputs=[box_output, class_output]
    )
    
    return model

# 自定义损失函数
class DetectionLoss(tf.keras.losses.Loss):
    def __init__(self, name='detection_loss'):
        super().__init__(name=name)
        self.huber = tf.keras.losses.Huber()
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()
        
    def call(self, y_true, y_pred):
        # 解析真实值
        true_box, true_class = y_true
        pred_box, pred_class = y_pred
        
        # 计算边界框损失
        box_loss = self.huber(true_box, pred_box)
        
        # 计算分类损失
        class_loss = self.crossentropy(true_class, pred_class)
        
        # 总损失
        return box_loss + class_loss

def convert_to_tflite(model, filename='garbage_detector.tflite'):
    # 转换为TFLite模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    # 保存模型
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite模型已保存为: {filename}")

def main():
    # 创建数据集
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_data = train_dataset.create_dataset()
    val_data = val_dataset.create_dataset()
    
    # 计算每个 epoch 的步数
    steps_per_epoch = len(train_dataset.data) // BATCH_SIZE
    validation_steps = len(val_dataset.data) // BATCH_SIZE
    
    # 创建模型（其他部分保持不变）
    model = create_model()
    
    # 编译模型
    losses = {
        'box_output': tf.keras.losses.Huber(),
        'class_output': tf.keras.losses.SparseCategoricalCrossentropy()
    }
    
    metrics = {
        'box_output': 'mse',
        'class_output': 'accuracy'
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    # 训练模型
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        steps_per_epoch=steps_per_epoch,      # 添加步数限制
        validation_steps=validation_steps,     # 添加验证步数限制
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
    )
    
    # 保存Keras模型
    model.save('garbage_detector.h5')
    
    # 转换为TFLite模型
    convert_to_tflite(model)

if __name__ == '__main__':
    main()
