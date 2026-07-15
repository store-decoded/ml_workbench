import tensorflow as tf
import numpy as np
import json

# Configuration
IMG_SIZE = (224, 224)
CHECKPOINT_PATH = './checkpoints_tl/mobilenetv2_transfer.keras'
TEST_IMAGES = [
    "./dataset/transfer_learning/val/croissant/sample1.jpg", 
    "./dataset/transfer_learning/val/baguette/sample2.jpg"
]

def load_and_preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)
    return img_array

if __name__ == '__main__':
    # 1. Load Class Names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    # 2. Load Trained Model
    print("Loading model...")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

    # 3. Predict
    print("\n--- Predictions ---")
    for path in TEST_IMAGES:
        try:
            # Preprocess
            img_tensor = load_and_preprocess_image(path)
            
            # Inference
            predictions = model.predict(img_tensor, verbose=0)
            
            # Get class index with highest probability
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            
            predicted_class = class_names[pred_idx]
            print(f"Image: {path.split('/')[-1]} -> Predicted: {predicted_class} (Confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Failed to process {path}: {e}")
