import tensorflow as tf

#save to file
#W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
#b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
#init = tf.initialize_all_variables()
#saver = tf.train.Saver()

#with tf.Session() as sess:
#    sess.run(init)
#    save_path = saver.save(sess, "save_net.ckpt")
#    print("Save to path", save_path)



#restore from save_net.ckpt
import numpy as np
W2 = tf.Variable(np.arange(6).reshape(2,3), dtype=tf.float32, name="weights")
b2 = tf.Variable(np.arange(3).reshape(1,3), dtype=tf.float32, name="biases")
#not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "save_net.ckpt") #Restore by name
    print sess.run(W2)
    print sess.run(b2)

