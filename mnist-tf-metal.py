import os
import tensorflow as tf
import tensorflow_datasets as tfds

# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='cpu') # Available options are 'cpu', 'gpu', and 'any'.

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '4' #if not hvd_utils.is_using_hvd() else str(hvd.size())

BATCH_SIZE = 1024
EPOCHS = 100
AUTOTUNE = tf.data.AUTOTUNE


def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            for gpu in gpus_used:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


set_gpu([0])


# load MNIST
print('\ndownloading mnist')
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalize images: 'unit8' -> 'float32'."""
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE, num_parallel_calls=AUTOTUNE)

print('\ncreate and compile model')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer="adam", #tf.keras.optimizers.legacy.SGD(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test, verbose=1)
