### MNIST Example Taken from https://www.tensorflow.org/datasets/keras_example


import tensorflow as tf
import tensorflow_datasets as tfds
import spaceray
from argparse import ArgumentParser
import os
from ray import tune


def model_train(config, extra_data_dir):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ### Note this is where Ray Tune is plugging in a config parameter for a trial
    ds_train = ds_train.batch(config['batch_size'])
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ### Note this is where Ray Tune is plugging in a config parameter for a trial
    ds_test = ds_test.batch(config['batch_size'])
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        ### Note this is where Ray Tune is plugging in a config parameter for a trial
        optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        ### Note this is where Ray Tune is plugging in a config parameter for a trial
        epochs=config['epochs'],
    )

    res = model.evaluate(ds_test)

    # in order to save the model we use the information provided in the extra data dir
    tf_model_path = os.path.join(extra_data_dir['results_dir'], 'tf_models')
    if not os.path.exists(tf_model_path):
        os.makedirs(tf_model_path)
    model.save(os.path.join(extra_data_dir['results_dir'], 'tf_models', 'tf_model' + tune.get_trial_name() + '.h5'))

    return res[1]


if __name__ == "__main__":
    parser = ArgumentParser("Start bi model tuning with hyperspace and resiliency testing, "
                            "specify output csv file name.")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("-t", "--trials")
    parser.add_argument("-j", "--json")
    parser.add_argument('-n', '--start_space')
    parser.add_argument('-p', '--project_name', default="hyper_sensitive")
    args = parser.parse_args()
    main = os.getcwd()
    results = args.out[:-4]
    results = os.path.join(main, results)
    spaceray.run_experiment(args, model_train, ray_dir="/tmp/", cpu=8,
                            start_space=int(args.start_space), mode="max", project_name=args.project_name,
                            group_name='benchmark', extra_data_dir={'results_dir': results})
