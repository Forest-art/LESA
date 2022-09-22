import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import os
import pickle
import cv2
from core.model_share_attention import AttentionClassifier
from core.utils import evaluate_k,Logger,get_compress_type,LearningRate,evaluate
from global_setting import dim_feature,NUS_WIDE_train_img_path,NUS_WIDE_test_img_path,NUS_WIDE_val_img_path,\
                                    NUS_WIDE_zs_n_iters,NFS_path,batch_size,NUS_WIDE_init_w2v,description,NUS_WIDE_signal_str

def parser(record):
    feature = {'img_id': tf.FixedLenFeature([], tf.string),
               'feature': tf.FixedLenFeature([], tf.string),
               'label_1k': tf.FixedLenFeature([], tf.string),
               'label_81': tf.FixedLenFeature([], tf.string)}

    parsed = tf.parse_single_example(record, feature)
    img_id = parsed['img_id']
    feature = tf.reshape(tf.decode_raw( parsed['feature'],tf.float32),dim_feature)
    label_1k = tf.decode_raw( parsed['label_1k'],tf.float32)
    label_81 = tf.decode_raw( parsed['label_81'],tf.float32)
    label_925 = tf.gather(label_1k,seen_cls_idx)
    labels_1006 = tf.concat([label_81,label_925],axis=0)
    return img_id,feature,label_1k,label_81,label_925,labels_1006



def load_image(img_path):
	print("Loading image")
	img = cv2.imread(img_path)
	img = cv2.resize(img, (224, 224))
	return img


def get_seen_unseen_classes(file_tag1k,file_tag81):
    with open(file_tag1k,"r") as file: 
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81,"r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array([i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx,unseen_cls_idx,tag1k,tag81





def grad_cam(input_feature, model, sess, predicted_class, layer_name, nb_classes):
    print("Setting gradients to 1 for target class and rest to 0")
    # Conv layer tensor [?,7,7,512]
    conv_layer = layer_name
    # [1000]-D tensor with target class index set to 1 and rest as 0
    one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
    print(zsl_stat)
    signal = tf.multiply(model.gzs_logits if zsl_stat=="gzsl" else model.zs_logits, one_hot)
    loss = tf.reduce_mean(signal)

    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))  
    output, grads_val, loss_v = sess.run([conv_layer, norm_grads, loss], feed_dict={model.features: input_feature})
    print(loss_v)
    output = output[0].reshape(14,14,-1)          # [14,14,512]
    grads_val = grads_val[0].reshape(14,14,-1)	 # [14,14,512]

    weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = resize(cam, (224,224))

    # Converting grayscale to 3-D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3,[1,1,3])

    return cam3





sess = tf.Session()

# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
learning_rate_phase_1 = 0.001
n_anneal = 10
file_tag1k = NFS_path + '../data/NUS_WIDE/NUS_WID_Tags/TagList1k.txt'
file_tag81 = NFS_path + '../data/NUS_WIDE/Concepts81.txt'
seen_cls_idx,unseen_cls_idx,tag1k,tag81=get_seen_unseen_classes(file_tag1k,file_tag81)
dataset_tst = tf.data.TFRecordDataset(NUS_WIDE_test_img_path,compression_type=get_compress_type(NUS_WIDE_test_img_path))
dataset_tst = dataset_tst.map(parser)
dataset_tst = dataset_tst.batch(1)
iterator_tst = dataset_tst.make_initializable_iterator()
(img_ids_tst,features_tst,labels_1k_tst,labels_81_tst,labels_925_tst,labels_1006_tst) = iterator_tst.get_next()
with open(NUS_WIDE_init_w2v,'rb') as infile:
    vecs_1k,vecs_81 = pickle.load(infile)
    vecs_925 = vecs_1k[seen_cls_idx]
n_classes = tag81.shape[0]
with tf.variable_scope(tf.get_variable_scope()):
    model = AttentionClassifier(vecs = vecs_925,unseen_vecs=vecs_81,T=10,trainable_vecs=False,lamb_att_dist=0.1,
                                lamb_att_global=0.001,lamb_att_span=0.01,dim_feature=dim_feature,is_batchnorm=True,is_separate_W_1=False)
    model._log('lr {}'.format(learning_rate_phase_1))
    model._log('n_iters {}'.format(NUS_WIDE_zs_n_iters))
    model._log('no shuffle')
    model._log('adaptive learning rate')
    model._log('n_anneal {}'.format(n_anneal))
    model._log(description)
    model._log('train_img: '+NUS_WIDE_train_img_path)
    model._log('test_img: '+NUS_WIDE_test_img_path)
    model._log('val_img: '+NUS_WIDE_val_img_path)
    model.build_model_rank(is_conv=False)
#_ = input('confirm??')
#%%
sess = tf.InteractiveSession()
#%%
model._log('adaptive learning rate')
lr = LearningRate(learning_rate_phase_1,sess,signal_strength=NUS_WIDE_signal_str,patient=5)
model._log('signal_str {}'.format(lr.signal_strength))
model._log('patient {}'.format(lr.patient))
optimizer = tf.train.RMSPropOptimizer(
      lr.get_lr(),
      0.9,  # decay
      0.9,  # momentum
      1.0   #rmsprop_epsilon
  )
grads = optimizer.compute_gradients(model.loss)
print('-'*30)
print('Decompose update ops')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.apply_gradients(grads)
print('-'*30)
#%%
tf.global_variables_initializer().run()
saver = tf.train.Saver()
#%%
tf.summary.scalar('batch_loss', model.loss)
for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

for grad, var in grads:
    tf.summary.histogram(var.op.name+'/gradient', grad)
    
summary_op = tf.summary.merge_all()
name = "release_zs_NUS_WIDE_log_GPU_7_1587185916d2570488"
save_path = NFS_path+'results/'+name
saver.restore(sess, save_path+'/model.ckpt')  

print("*"*100)
# print("\nFeedforwarding")
# prob = sess.run(vgg.probs, feed_dict={vgg.imgs: x})[0]
# preds = (np.argsort(prob)[::-1])[0:5]
# print('\nTop 5 classes are')
# for p in preds:
#     print(class_names[p], prob[p])

# # Target class
# predicted_class = preds[0]

zsl_stat = "zsl"   ## zsl or gzsl

# Target layer for visualization
layer_name = model.gradcam
print(layer_name)
# Number of output classes of model being used
nb_classes = 1006 if zsl_stat=="gzsl" else 81
sess.run(iterator_tst.initializer)

for i in range(100000):
    img_ids_v,features_v,labels_v = sess.run([img_ids_tst,features_tst,labels_1006_tst if zsl_stat=="gzsl" else labels_81_tst])
    if i % 1000 == 0:
        img_name = img_ids_v[0].decode('UTF-8').split('/')[-2] + '/' + img_ids_v[0].decode('UTF-8').split('/')[-1]
        img = load_image("/opt/disk/luxc/Dataset/Flickr/" + img_name)
        lab_idx = np.where(labels_v[0]==1)
        print(img_ids_v, lab_idx)
        for j, idx in enumerate(lab_idx[0]):
            if j > 2:
                break
            predicted_class = idx
            # predicted_class = 42
            cam3 = grad_cam(features_v, model, sess, predicted_class, layer_name, nb_classes)
            print(tag1k[predicted_class])
            cam_heatmap = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
            # cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

            img = img.astype(float)
            img /= img.max()
            img = np.uint8(255*img)
            # Superimposing the visualization with the image.
            # new_img = img+3*cam3
            # new_img /= new_img.max()
            new_image = cv2.addWeighted(img,0.3,cam_heatmap,0.7,0)
            # Display and save
            os.makedirs("HeatMap/{}/".format(zsl_stat), exist_ok=True)
            tag = tag1k if zsl_stat=="gzsl" else tag81
            cv2.imwrite("HeatMap/{}/".format(zsl_stat) + img_name.replace("/","_").split(".")[0] + "_" + tag[predicted_class] + ".jpg", new_image)
            # io.imshow(new_image)
            # plt.show()


# %%
