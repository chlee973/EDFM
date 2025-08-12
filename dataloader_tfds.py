import tensorflow as tf
import tensorflow_datasets as tfds

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
SEED = 42


def build_dataloader(batch_size):
    tf.random.set_seed(SEED)  # Set the random seed for reproducibility.
    train_ds: tf.data.Dataset = tfds.load("cifar10", split="train")
    test_ds: tf.data.Dataset = tfds.load("cifar10", split="test")

    mean = tf.constant(PIXEL_MEAN, shape=(1, 1, 3), dtype=tf.float32)
    std = tf.constant(PIXEL_STD, shape=(1, 1, 3), dtype=tf.float32)

    def preprocess_train(sample):
        image = sample
        image = tf.cast(image["image"], tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.pad(image, paddings=[[4, 4], [4, 4], [0, 0]])
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = (image - mean) / std
        return {"image": image, "label": sample["label"]}

    def preprocess_eval(sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        image = (image - mean) / std
        return {"image": image, "label": sample["label"]}

    train_ds = train_ds.shuffle(50000).repeat()
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds
