import tensorflow as tf
import json
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
TRAIN_DIR = './dataset/train'

def build_model(num_classes):
    # 1. Load Pre-trained Model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Freeze Base Layers
    base_model.trainable = False

    # 3. Build new Classification Head
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # MobileNetV2 expects pixel values in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    # 1. Data Preparation
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    class_names = train_dataset.class_names

    # Save class names for prediction script
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Initialize and Compile Model
    model = build_model(num_classes=len(class_names))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # 3. Checkpoint Setup
    os.makedirs('checkpoints_tl', exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints_tl/mobilenetv2_transfer.keras',
        save_best_only=True,
        monitor='loss'
    )

    # 4. Train
    print("Starting Transfer Learning Training...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback]
    )
