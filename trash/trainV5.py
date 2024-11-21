import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 设置内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 获取数据目录的绝对路径
data_dir = os.path.abspath('train_data/')

# 定义图像参数
IMG_HEIGHT, IMG_WIDTH = 180, 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# 增强的数据增强层
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomCrop(160, 160),
    layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
])

def prepare_dataset(ds, cache=True):
    """数据集优化函数"""
    if cache:
        ds = ds.cache()
    ds = ds.shuffle(2000)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# 加载数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 保存类别名称
class_names = train_ds.class_names

# 优化数据集
train_ds = prepare_dataset(train_ds)
val_ds = prepare_dataset(val_ds)

# 打印数据集信息
print("训练集大小:", tf.data.experimental.cardinality(train_ds).numpy())
print("验证集大小:", tf.data.experimental.cardinality(val_ds).numpy())
print("类别:", class_names)

def create_model(num_classes):
    """创建模型函数"""
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False

    model = models.Sequential([
        data_augmentation,
        layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def unfreeze_model(model, num_layers):
    """解冻模型层函数"""
    base_model = model.layers[2]  # 获取基础模型层
    # 冻结底部层
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    # 解冻顶部层
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    return model

# 创建模型
model = create_model(len(class_names))

# 定义学习率调度器
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 定义回调函数
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        min_delta=0.001
    ),
    ModelCheckpoint(
        'best_model_initial.keras',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv')
]

# 打印模型结构
model.summary()

# 初始训练
print("开始初始训练...")
initial_epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=callbacks
)

# 分阶段微调
unfreeze_layers = [30, 50, 100]  # 分三阶段解冻
fine_tune_lr = 5e-6  # 初始微调学习率

for i, num_layers in enumerate(unfreeze_layers):
    print(f"\n开始第 {i+1} 阶段微调 (解冻后 {num_layers} 层)...")
    
    model = unfreeze_model(model, num_layers)
    
    # 为微调阶段编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 定义该阶段的回调函数
    fine_tune_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            min_delta=0.001
        ),
        ModelCheckpoint(
            f'best_model_finetuned_stage_{i+1}.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # 训练该阶段
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=fine_tune_callbacks
    )
    
    # 降低下一阶段的学习率
    fine_tune_lr *= 0.5

# 保存最终模型
model.save('garbage_classifier_final.keras')

# 转换为 TFLite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存 TFLite 模型
with open('garbage_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend(loc='lower right')
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# 绘制训练历史
plot_training_history(history)