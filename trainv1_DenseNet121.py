import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
# GPU 配置
def configure_gpu():
    # 获取可用的GPU列表
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 如果有多个GPU，可以选择使用特定的GPU
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"找到 {len(gpus)} 个物理GPU, {len(logical_gpus)} 个逻辑GPU")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    else:
        print("未找到可用的GPU，将使用CPU训练")

# 在代码开始时调用GPU配置
configure_gpu()
# 数据预处理参数
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 40

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
                
        self.num_samples = len(self.data)
        print(f"加载了 {self.num_samples} 个样本")
    
    def preprocess_image(self, img_path):
        # 读取和预处理图片
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            # 返回一个空白图片作为替代
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        img = img.astype(np.float32) / 255.0
        
        if self.is_training:
            # 数据增强
            if np.random.random() > 0.5:
                img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            
        return img
    
    def create_dataset(self):
        def generator():
            while True:  # 添加无限循环
                indices = np.random.permutation(len(self.data))
                for idx in indices:
                    img_path, label = self.data[idx]
                    yield self.preprocess_image(img_path), label
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # 设置批处理
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def create_model():
    base_model = tf.keras.applications.DenseNet121(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def convert_to_tflite(model, dataset, filename='garbage_classifier.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite模型已保存为: {filename}")

def main():
    # 创建数据集
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_data = train_dataset.create_dataset()
    val_data = val_dataset.create_dataset()
    
    # 计算steps_per_epoch
    steps_per_epoch = train_dataset.num_samples // BATCH_SIZE
    validation_steps = val_dataset.num_samples // BATCH_SIZE
    
    # 创建和编译模型
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练模型
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
    )
    
    # 保存Keras模型
    model.save('garbage_classifier.h5')
    
    # 转换为TFLite模型
    convert_to_tflite(model, val_data)

if __name__ == '__main__':
    main()
