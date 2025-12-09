import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def main():
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "archive", "asl_alphabet_train")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Config
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 5  # Transfer learning is fast, 5 epochs are enough
    
    print(f"[INFO] Training data: {train_dir}")
    print("[INFO] Loading MobileNetV2...")

    # Data Generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.1,
        rotation_range=20,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Transfer Learning Model (MobileNetV2)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(29, activation='softmax')(x)  # 29 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training
    print("[INFO] Starting training...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    # Save Model
    model_save_path = os.path.join(models_dir, "asl_detection_model.h5")
    model.save(model_save_path)
    print(f"[SUCCESS] Model saved to: {model_save_path}")

    # Save Class Indices (important for mapping 0 -> 'A')
    indices_path = os.path.join(models_dir, "class_indices.json")
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(indices_path, "w") as f:
        json.dump(idx_to_class, f, indent=4)
    print(f"[SUCCESS] Class indices saved to: {indices_path}")

if __name__ == "__main__":
    main()
