import tensorflow as tf
g = tf.Graph() 
with g.as_default() as g: 
    tf.train.import_meta_graph('./results/release_zs_NUS_WIDE_log_GPU_7_1587185916d2570488/model.ckpt.meta') 
 
with tf.Session(graph=g) as sess: 
    file_writer = tf.summary.FileWriter(logdir='./results/release_zs_NUS_WIDE_log_GPU_7_1587185916d2570488/', graph=g)