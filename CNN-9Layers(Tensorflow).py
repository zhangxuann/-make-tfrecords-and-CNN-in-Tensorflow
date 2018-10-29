#这是一个9层的CNN，4个卷积层，3个池化层，2个全连接层，解决关于人脸识别的二分类问题。

#导入TensorFlow和相应的库、函数
import tensorflow as tf
import numpy as np
import time

#定义最大训练次数max_steps,及batch_size和文件路径
max_steps=3000
batch_size=128
data_path='zx/tf/face_recog/face.tfrecords'

#初始化weight的函数，加正则L2的损失；weight_loss收集到“losses”.
def variable_with_weight_loss(shape,stddev,wl):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection(weight_loss,name='losses')
    return var		
	
#读取训练和测试图片
images_train,labels_train=input(data_path=data_path,batch_size=batch_size)
images_test,labels_test=input(test_data=True,data_path=data_path,batch_size=batch_size)
	
#占位符，定义图片和标签的字符类型及数据形式	
image_holder=tf.placeholder(tf.float32,[batch_size,24,24,3]) #
label_holder=tf.placeholder(tf.int32,[batch_size]) #

#定义第一个卷积层conv1和池化层pool1，在conv1后有ReLU,
weight1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,wl=0.0)
kernel1=tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
conv1=tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME' )

#定义conv2和pool2，类似于前者
weight2=variable_with_weight_loss(shape=[5,5,64,128],stddev=5e-2,wl=0.0)
kernel2=tf.nn.conv2d(pool1,weight2,[1,1,1,1],padding='SAME')
bias2=tf.Variable(tf.constant(0.1,shape=[128]))
conv2=tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME' )

#定义conv3和pool3，类似于前者，只是卷积核大小、padding不同
weight3=variable_with_weight_loss(shape=[3,3,128,256],stddev=5e-2,wl=0.0)
kernel3=tf.nn.conv2d(pool2,weight3,[1,1,1,1],padding='VALID')
bias3=tf.Variable(tf.constant(0.1,shape=[256]))
conv3=tf.nn.relu(tf.nn.bias_add(kernel3,bias3))
pool3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID' )

#定义conv4和ReLU，
weight4=variable_with_weight_loss(shape=[2,2,256,512],stddev=5e-2,wl=0.0)
kernel4=tf.nn.conv2d(pool3,weight4,[1,1,1,1],padding='VALID')
bias4=tf.Variable(tf.constant(0.1,shape=[512]))
conv4=tf.nn.relu(tf.nn.bias_add(kernel4,bias4))

#使用tf.reshape将conv4的输出结果变成一维向量，节点数变为256，构造第一个全连接层local1
reshape=tf.reshape(conv4,[batch_size,-1])
dim=reshape.get_shape()[1].value
weight5=variable_with_weight_loss(shape=[dim,256],stddev=0.04,wl=0.004)
bias5=tf.Variable(tf.constant(0.1,shape=[256]))
local1=tf.nn.relu(tf.matmul(reshape,weight5)+bias5)

#最后一层logits,分为2个类输出
weight6=variable_with_weight_loss(shape=[256,2],stddev=1/256.0,wl=0.0)
bias6=tf.Variable(tf.constant(0.1,shape=[2]))
logits=tf.add(tf.matmul(local1,weight6)+bias6)

#构造损失loss函数，将softmax和cross_entroy_loss结合，tf.add_n求和所有loss，得到‘total_loss’ 
def loss(logits,labels):
    labels=tf.casst(labels,tf.int64)
	cross_entroy=tf.nn.sparse_softmax_cross_entroy_with_logits(logits=logits,labels=labels,
	                 name='cross_entroy_per_example')
	cross_entroy_mean=tf.reduce_mean(cross_entroy,name='cross_entroy')
	tf.add_to_collection('losses',cross_entroy_mean)
	return tf.add_n(tf.get_collection('losses'),name='total_loss')

#将最后一层logits节点传入loss函数，优化器是AdamOptimizer，输出分数最高的那一类的准确率（k=1）
loss=loss(logits,label_holder)
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op=tf.nn.in_top_k(logits,label_holder,1)

#创建默认会话，全部参数初始化
sess=tf.InteractiveSession()
tf.global_variable_initializer().run()

#启动队列，开始训练，TF一般是以队列形式读入并处理数据
tf.train.start_queue_runners()

#每一个step的训练，先用sess.run()获取数据，计算每一个step所需时间，每隔10个step，输出当前的loss、
# 每秒所训练的样本数、每个batch所用的时间。
for step in range(max_steps):
    start_time=time.time()
	image_batch,label_batch
    _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
	duration=time.time()-start_time
	if step%10==0:
	    examples_per_sec=batch_size/duration
		sec_per_batch=float(duration)
		format_str=('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
		print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
		

#计算模型在测试集上的准确率		
num_examples=10000
import math
num_iter=int(math.ceil(num_examples/batch_size))
true_count=0
total_sample_count=num_iter*batch_size
step=0
while step<num_iter:
    image_batch,label_batch=sess.run([images_test,labels_test])
	predictions=sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
	true_count+=np.sum(predictions)
	step+=1

#输出准确率。	
precision=true_count/total_sample_count
print('precision @ 1=%.3f' %precision)
