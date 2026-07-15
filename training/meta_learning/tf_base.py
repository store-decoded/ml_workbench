import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, applications
from PIL import Image

# --- 1. Prototypical Network Model ---
class ProtoNet(tf.keras.Model):
    def __init__(self, n_way=5, k_shot=5, query_size=5):
        super(ProtoNet, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        
        # Backbone (MobileNetV2 is lightweight and standard for small image tasks)
        base_model = applications.MobileNetV2(
            input_shape=(84, 84, 3), 
            include_top=False, 
            weights=None
        )
        self.encoder = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D()
        ])
        
        # Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.encoder(inputs)

    def train_step(self, data):
        # Unpack data (batch_size is 1 episode)
        images, _ = data[0] # Images shape: [1, N*(K+Q), 84, 84, 3]
        images = images[0]  # Remove batch dim: [N*(K+Q), 84, 84, 3]
        
        with tf.GradientTape() as tape:
            embeddings = self.encoder(images, training=True)
            emb_dim = tf.shape(embeddings)[-1]
            
            # Reshape to [N, K+Q, Dim]
            embeddings = tf.reshape(embeddings, [self.n_way, self.k_shot + self.query_size, emb_dim])
            
            # Split into support and query
            support = embeddings[:, :self.k_shot, :] # [N, K, Dim]
            query = embeddings[:, self.k_shot:, :]   # [N, Q, Dim]
            
            # Calculate Prototypes (mean of support set)
            prototypes = tf.reduce_mean(support, axis=1) # [N, Dim]
            
            # Flatten queries: [N*Q, Dim]
            query_flat = tf.reshape(query, [self.n_way * self.query_size, emb_dim])
            
            # Calculate Euclidean distance using broadcasting
            # query_ext: [N*Q, 1, Dim], proto_ext: [1, N, Dim] => diff: [N*Q, N, Dim]
            query_ext = tf.expand_dims(query_flat, 1)
            proto_ext = tf.expand_dims(prototypes, 0)
            distances = tf.norm(query_ext - proto_ext, axis=2) # [N*Q, N]
            
            # Generate targets: [0,0.., 1,1.., N,N..]
            targets = tf.repeat(tf.range(self.n_way), self.query_size)
            
            # Loss is sparse categorical crossentropy on negative distances
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets, -distances, from_logits=True
            )
            loss = tf.reduce_mean(loss)
            
        # Gradients & Backprop
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Accuracy
        preds = tf.argmin(distances, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, targets), tf.float32))
        
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

# --- 2. Dummy Episodic Generator ---
class EpisodicGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_dir, n_way, k_shot, q_size, episodes_per_epoch=100):
        self.n_way = n_way
        self.total_per_class = k_shot + q_size
        self.episodes = episodes_per_epoch
        
        self.class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                           if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        return self.episodes
        
    def __getitem__(self, idx):
        sampled_classes = random.sample(self.class_dirs, self.n_way)
        episode_imgs = []
        episode_lbls = []
        
        for c_idx, c_dir in enumerate(sampled_classes):
            all_imgs = os.listdir(c_dir)
            sampled_imgs = random.sample(all_imgs, self.total_per_class)
            for img_name in sampled_imgs:
                img_path = os.path.join(c_dir, img_name)
                img = Image.open(img_path).convert('RGB').resize((84, 84))
                img_arr = np.array(img) / 255.0 # Normalize to [0,1]
                
                # Standardize to ImageNet means/stds
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_arr = (img_arr - mean) / std
                
                episode_imgs.append(img_arr)
                episode_lbls.append(c_idx)
                
        # Return batched shape: [1, N*(K+Q), 84, 84, 3]
        batch_x = np.expand_dims(np.array(episode_imgs, dtype=np.float32), axis=0)
        batch_y = np.expand_dims(np.array(episode_lbls, dtype=np.int32), axis=0)
        return batch_x, batch_y

if __name__ == '__main__':
    N_WAY = 5
    K_SHOT = 5
    Q_SIZE = 5
    
    # Load Data
    train_gen = EpisodicGenerator('./datasets/meta_learning/train', N_WAY, K_SHOT, Q_SIZE)
    
    # Initialize Model
    model = ProtoNet(n_way=N_WAY, k_shot=K_SHOT, query_size=Q_SIZE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    # Checkpoint (Save weights only since it's a subclassed model)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./resources/checkpoints/protonet_tf',
        save_weights_only=True,
        save_best_only=True,
        monitor='loss'
    )
    
    # Train
    model.fit(train_gen, epochs=10, callbacks=[checkpoint_callback])
