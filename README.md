# -make-tfrecords-datasets-for-Tensorflow

#   主要是第一个项目，深度学习用于人脸识别。拿到原始数据，用标注程序标记人脸照片，然后将标注的人脸照片制作成tfrecords数据集。在TensorFlow平台上实现CNN用于人脸识别，这里包括基本的CNN、VGGNet、GoogLeNet、ResNet等网络，训练人脸数据集，测试网络性能。相对于原始的VGGNet、GoogLeNet和ResNet，我主要做了：1，加深网络层数和拓宽网络；2，改变卷积核、步长、损失函数、batch_size、迭代次数等参数；3，增加或改变某些网络结构，比如inception modules，池化层，dropout层等。通过这些措施，训练和调优上述网络的变体。这里做的改变，和在caffe平台上的CNN实验类似，主要是熟练了TensorFlow平台和深入理解了深度神经网络的原理和编程实现。
#   这里只展示一个基础的CNN网络训练代码。读取tfrecords数据后，进入网络开始训练，源代码见CNN-9Layers-TF。

# make.tfrecords.py:制作tfrecords数据集：TensorFlow可以处理的数据集格式之一是tfrecords，主要用到Python和TensorFlow框架中的函数或库。这里包含制作tfrecords数据集、验证是否制作成功以及将数据读入深度神经网络三部分的程序，编程实现比较多样，这里仅是我编程实现的过程,python编写。

# 第二部分是CNN-9Layers-tensorflow的编程实现。
