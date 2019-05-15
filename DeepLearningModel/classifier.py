import argparse
import tensorflow as tf
import read_data

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', default=100, type=int,
                   help='batch size')
parse.add_argument('--train_steps', default=1000, type=int,
                   help='training steps')


def main(argv):
    args = parse.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = read_data.load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=2,
        # optimizer=tf.train.AdamOptimizer(
        #     learning_rate=0.01
        # ),
        model_dir='model'
    )

    print("Begin Training")
    classifier.train(
        input_fn=lambda: read_data.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps
    )

    print("End Training")

    print("Begin Evaluate")
    eval_result = classifier.evaluate(
        input_fn=lambda: read_data.eval_input_fn(test_x, test_y,
                                                 args.batch_size)
    )
    print("End Evaluate")

    print(eval_result)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # expected = ['similar', 'dissimilar']
    #
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    # predictions = classifier.predict(
    #     input_fn=lambda: read_data.eval_input_fn(predict_x,
    #                                              labels=None,
    #                                              batch_size=args.batch_size)
    # )
    # template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'
    # for pre_dict, expec in zip(predictions, expected):
    #     class_id = pre_dict['class_ids'][0]
    #     probability = pre_dict['probabilities'][class_id]
    #     print(template.format(read_data.SAME[class_id],
    #                           100 * probability, expec))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
