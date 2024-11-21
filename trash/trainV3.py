import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 获取 train_data/ 目录的绝对路径
data_dir = os.path.abspath('train_data/')

# 加载数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# 打印数据集信息
print("训练集大小:", tf.data.experimental.cardinality(train_ds).numpy())
print("验证集大小:", tf.data.experimental.cardinality(val_ds).numpy())

# 定义图像高度和宽度
img_height, img_width = 180, 180

# 数据增强和归一化
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 构建模型
class_names = train_ds.class_names

base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # 冻结预训练层

model = models.Sequential([
    data_augmentation,
    layers.Lambda(preprocess_input),  # 使用 Lambda 层进行预处理
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),  # 添加 Dropout 层
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 定义回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-4),
    ModelCheckpoint('best_model_initial.keras', save_best_only=True)
]

# 初始训练模型
epochs = 180
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# 微调模型
base_model.trainable = True  # 解冻整个预训练模型

# 选择从哪一层开始解冻
fine_tune_at = 100  # 根据模型结构调整

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# 使用较低的学习率重新编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定义微调阶段的回调函数
fine_tune_callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_model_finetuned.keras', save_best_only=True)
]

# 继续训练模型
fine_tune_epochs = 40
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=fine_tune_callbacks
)

# 合并历史数据
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epochs_range = range(len(acc))  # acc 的长度为实际训练的 epoch 数量



# 保存最佳模型（如果需要）
model.save('garbage_classifier_final.keras')

# 转换为 TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存 TensorFlow Lite 模型
with open('garbage_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

# 打印各列表长度以确认一致性
print("总训练轮次:", len(acc))
print("epochs_range 长度:", len(epochs_range))
print("训练准确率长度:", len(acc))
print("验证准确率长度:", len(val_acc))
print("训练损失长度:", len(loss))
print("验证损失长度:", len(val_loss))
plt.figure(figsize=(12, 4))

# 训练与验证准确率
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.legend(loc='lower right')
plt.title('训练与验证准确率')

# 训练与验证损失
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练损失')
plt.plot(epochs_range, val_loss, label='验证损失')
plt.legend(loc='upper right')
plt.title('训练与验证损失')

plt.show()