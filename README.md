# Grafik-cart2
Grafik cart2
# Code for Multiple GPU Connections with Data Parallelism
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Open a strategy scope
    with strategy.scope():
        # Create and compile a model
        model = Sequential([
            Dense(128, input_shape=(784,), activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess your data (replace this with your actual data loading code)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255.0
    y_train = y_train.astype('float32')

    # Train the model
    model.fit(x_train, y_train, epochs=3, batch_size=64)

else:
    print("No GPUs available.")
