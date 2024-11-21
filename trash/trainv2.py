import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

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
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
    layers.RandomBrightness(0.2),
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
    layers.Rescaling(1./127.5, offset=-1),  # 替代 Lambda 层进行预处理
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 设置回调函数
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=10,
                               restore_best_weights=True)

lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=3,
                                 factor=0.5,
                                 min_lr=1e-6)

# 训练模型
epochs = 60
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

# 绘制训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.legend(loc='lower right')
plt.title('训练与验证准确率')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练损失')
plt.plot(epochs_range, val_loss, label='验证损失')
plt.legend(loc='upper right')
plt.title('训练与验证损失')
plt.show()

# 保存最佳模型
model.save('garbage_classifier_final.keras')

# 转换为 TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存 TensorFlow Lite 模型
with open('garbage_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

# 微调预训练模型（可选）
# 如果在冻结预训练层后模型性能仍不理想，可以尝试解冻部分层进行微调
base_model.trainable = True

# 只解冻最后几个卷积块
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 重新编译模型，使用较低的学习率
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 继续训练
fine_epochs = 30
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_epochs,
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

# 再次绘制训练过程
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.legend(loc='lower right')
plt.title('训练与验证准确率 (微调后)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练损失')
plt.plot(epochs_range, val_loss, label='验证损失')
plt.legend(loc='upper right')
plt.title('训练与验证损失 (微调后)')
plt.show()

# 最终评估模型
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print(f'最终验证集准确率: {val_accuracy * 100:.2f}%')
