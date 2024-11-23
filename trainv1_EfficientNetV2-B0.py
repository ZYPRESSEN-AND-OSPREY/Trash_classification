import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import datetime

# GPU 配置
def configure_gpu():
    # 获取可用的GPU列表
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
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
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        img = img.astype(np.float32) / 255.0
        
        if self.is_training:
            # 基础数据增强
            if np.random.random() > 0.5:
                img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            
        return img
    
    def create_dataset(self):
        def generator():
            while True:
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
        
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def create_model():
    # 移除混合精度策略
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 优化的模型结构
    base_model.trainable = True
    # 冻结前面2/3的层
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(num_layers*2/3)]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2048, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
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

def create_callbacks(model_name="garbage_classifier"):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return [
        # TensorBoard回调
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        # 提前停止
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        # 学习率调度
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # 模型检查点 - 修改文件扩展名为.keras
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # 训练日志
        tf.keras.callbacks.CSVLogger(f'{model_name}_training_log.csv')
    ]

def create_optimizer():
    # 使用简单的指数衰减
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    return optimizer
def main():
    # 创建数据集
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_data = train_dataset.create_dataset()
    val_data = val_dataset.create_dataset()
    
    # 计算steps_per_epoch
    steps_per_epoch = train_dataset.num_samples // BATCH_SIZE
    validation_steps = val_dataset.num_samples // BATCH_SIZE
    
    # 创建模型
    model = create_model()
    
    # 编译模型
    model.compile(
        optimizer=create_optimizer(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # 训练模型
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=90,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # 保存最终模型 - 修改文件扩展名为.keras
    model.save('garbage_classifier_final.keras')
    
    # 转换为TFLite模型
    convert_to_tflite(model, val_data)
if __name__ == '__main__':
    main()
