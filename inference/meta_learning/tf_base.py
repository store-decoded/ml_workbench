import os
import numpy as np
import tensorflow as tf
from PIL import Image
from training.meta_learning.tf_base import ProtoNet 


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((84, 84))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_arr = (img_arr - mean) / std
    return img_arr

def predict_single_image(checkpoint_path, support_dir, query_image_path, k_shot=5):
    # 1. Initialize Model and Load Weights
    # Determine n_way dynamically based on evaluation folders
    class_names = sorted([d for d in os.listdir(support_dir) 
                          if os.path.isdir(os.path.join(support_dir, d))])
    n_way = len(class_names)
    
    model = ProtoNet(n_way=n_way, k_shot=k_shot, query_size=1)
    # Perform a dummy forward pass to build the model variables before loading weights
    model(tf.zeros((1, 84, 84, 3))) 
    model.load_weights(checkpoint_path).expect_partial()
    
    # 2. Build Support Set
    support_images = []
    support_labels = []
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(support_dir, class_name)
        images = os.listdir(class_dir)[:k_shot]
        
        for img_name in images:
            img_arr = preprocess_image(os.path.join(class_dir, img_name))
            support_images.append(img_arr)
            support_labels.append(label_idx)
            
    support_tensor = tf.convert_to_tensor(support_images) # [N*K, 84, 84, 3]
    
    # 3. Load Query Image
    q_img_arr = preprocess_image(query_image_path)
    query_tensor = tf.expand_dims(tf.convert_to_tensor(q_img_arr), axis=0) # [1, 84, 84, 3]
    
    # 4. Inference
    support_emb = model.encoder(support_tensor, training=False) # [N*K, Dim]
    query_emb = model.encoder(query_tensor, training=False)     # [1, Dim]
    
    # Compute Prototypes manually by averaging per class
    support_labels = tf.convert_to_tensor(support_labels)
    prototypes = []
    for c in range(n_way):
        class_embs = tf.boolean_mask(support_emb, tf.equal(support_labels, c))
        prototypes.append(tf.reduce_mean(class_embs, axis=0))
    prototypes = tf.stack(prototypes) # [N, Dim]
    
    # Compute Distance: $d(Q, P)$
    query_ext = tf.expand_dims(query_emb, 1) # [1, 1, Dim]
    proto_ext = tf.expand_dims(prototypes, 0) # [1, N, Dim]
    distances = tf.norm(query_ext - proto_ext, axis=2) # [1, N]
    
    # Prediction
    pred_idx = tf.argmin(distances, axis=1)[0].numpy()
    
    print(f"Predicted Class: {class_names[pred_idx]}")
    return class_names[pred_idx]

if __name__ == '__main__':
    CHECKPOINT = "./resources/checkpoints/protonet_tf"
    SUPPORT_DIR = "./datasets/meta_learning/eval" 
    QUERY_IMAGE = "./datasets/meta_learning/eval/class_A/sample_test.jpg"
    
    predict_single_image(CHECKPOINT, SUPPORT_DIR, QUERY_IMAGE)
