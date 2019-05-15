import tensorflow as tf


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=3)

model_file = tf.train.latest_checkpoint('model/')

saver.restore(sess, model_file)
