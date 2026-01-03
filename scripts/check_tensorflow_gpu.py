
import sys

def check_tensorflow_gpu():
    try:
        import tensorflow as tf
    except ImportError as e:
        print(f"FAILURE: TensorFlow not installed. Error: {e}")
        print("Install with: pip install tensorflow")
        return False

    print(f"TensorFlow version: {tf.__version__}")

    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")

    if not gpus:
        print("WARNING: No GPUs detected. TensorFlow will run on CPU only.")
        print("Possible reasons:")
        print("  - No NVIDIA GPU installed")
        print("  - CUDA toolkit not installed")
        print("  - cuDNN not installed")
        print("  - GPU not visible to TensorFlow")
        return False

    # Print GPU details
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"  Details: {gpu_details}")
        except Exception:
            pass

    # Test GPU computation
    try:
        print("\nAttempting simple GPU computation...")

        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Simple computation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            result = c.numpy()

        print(f"GPU computation successful! Result:\n{result}")

        # Test with a simple model
        print("\nAttempting to train a simple model on GPU...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Dummy data
        import numpy as np
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)

        # Train for 1 epoch as a test
        model.fit(X, y, epochs=1, verbose=0)

        print("SUCCESS: TensorFlow is properly configured for GPU training!")
        print(f"\nCUDA Version (built with): {tf.sysconfig.get_build_info().get('cuda_version', 'unknown')}")
        print(f"cuDNN Version (built with): {tf.sysconfig.get_build_info().get('cudnn_version', 'unknown')}")

        return True

    except Exception as e:
        print(f"FAILURE: GPU computation failed. Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure CUDA toolkit is installed and compatible with TensorFlow")
        print("  - Ensure cuDNN is installed and compatible with TensorFlow")
        print(f"  - Check TensorFlow GPU compatibility at: https://www.tensorflow.org/install/source#gpu")
        return False

if __name__ == "__main__":
    if check_tensorflow_gpu():
        sys.exit(0)
    else:
        sys.exit(1)
