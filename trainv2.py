import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 40

class GarbageDataset:
    def __init__(self, root_dir, txt_file, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        with open(os.path.join(root_dir, txt_file), 'r') as f:
            self.data = []
            for line in f:
                img_path, label = line.strip().split()
                img_path = img_path.lstrip('./')
                self.data.append((img_path, int(label)))
        print(f"加载了 {len(self.data)} 个样本")
    
    def preprocess_image(self, img_path):
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0
    
    def create_dataset(self):
        def generator():
            np.random.shuffle(self.data)
            for img_path, label in self.data:
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
        if self.is_training:
            dataset = dataset.repeat()
        return dataset

def create_model():
    # 使用EfficientNetB0作为backbone
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 微调策略：只训练最后30层
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # 多层特征提取器
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

def get_callbacks():
    return [
        # 提前停止
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率调整
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

def main():
    steps_per_epoch = 8869 // BATCH_SIZE
    
    train_dataset = GarbageDataset('garbage', 'train.txt', is_training=True)
    val_dataset = GarbageDataset('garbage', 'validate.txt', is_training=False)
    
    train_data = train_dataset.create_dataset()
    val_data = val_dataset.create_dataset()
    
    model = create_model()
    
    # 使用余弦衰减学习率
    initial_learning_rate = 0.001
    decay_steps = 1000
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps
    )
    
    # 使用AdamW优化器
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练模型
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        callbacks=get_callbacks()
    )
    
    # 保存最终模型
    model.save('final_model.keras')

if __name__ == '__main__':
    main()