"""制作tfrecords数据集：TensorFlow可以处理的数据集格式之一是tfrecords，主要用到Python和TensorFlow框架中的函数或库。
这里包含制作tfrecords数据集、验证是否制作成功以及将数据读入深度神经网络三部分的程序，编程实现比较多样，这里仅是我编程实现的过程。
一，制作tfrecords数据集"""
import tensorflow as tf #导入TensorFlow，并命名为tf
import os
#生成tfrecords的函数

def generate_tfrecords(input_filename, output_filename):  
    print("Start to convert {} to {}".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename) #写入要输出文件

    for line in open(input_filename, "r"): #输入文件转成列表
        data = line.split(",")
        label = float(data[2])             #标签数（或特征数）是0和1   
        features = [float(i) for i in data[:1]]

        example = tf.train.Example(features=tf.train.Features(feature=
        "label":
         tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "features":
         tf.train.Feature(float_list=tf.train.FloatList(value=features))))

    writer.write(example.SerializeToString())

    writer.close()
    print("Successfully convert {} to {}".format(input_filename,output_filename))
    
#主函数调用generate_tfrecords()，生成tfrecords数据集
def main():
    current_path = os.getcwd()
    for filename in os.listdir(current_path):
        if filename.startswith("") and filename.endswith(".csv"):
            generate_tfrecords(filename, filename + ".tfrecords")


if __name__ == "__main__":
    main()

"""二，验证tfrecords数据集是否制作成功"""
Import tensorflow as tf
Import os
def print_tfrecords(input_filename):
    max_print_number = 100
  current_print_number = 0

for serialized_example in tf.python_io.tf_record_iterator(input_filename):
    # 获得序列样本
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature["label"].float_list.value
    features = example.features.feature["features"].float_list.value
    print("Number: {}, label: {}, features: {}".format(current_print_number, label, features))


    # 获取所有的样本后，退出。
    current_print_number += 1
    if current_print_number > max_print_number:
        exit()


#运行函数，输出input_filename
def main():
    current_path = os.getcwd()
  tfrecords_file_name = "filename.csv.tfrecords"
  input_filename = os.path.join(current_path, tfrecords_file_name)
    print_tfrecords(input_filename)


if __name__ == "__main__":
    main()

"""三，读入神经网络:TensorFlow读取tfrecords文件时，以队列形式分批读入，CNN或RNN也会分批处理数据。下面是以文件队列形式读入神经网络。"""

import tensorflow as tf
import math
import os
import numpy as np
#定义读取tfrecords数据集的函数
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
#解析数据和标签
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([FEATURE_SIZE], tf.float32),
         })

    label = features["label"]
    features = features["features"]

    return label, features

#定义主函数，调用上述函数读取tfrecords文件
def main(_):
     #tfrecords文件，分为train和test两部分 
     filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once("train.tfrecords"))    

     label, features = read_and_decode(filename_queue)
     #随机散列batch
     batch_labels, batch_features = tf.train.shuffle_batch(
                                      [label, features],batch_size=batch_size)
    
     filename_queue = tf.test.string_input_producer(
                tf.test.match_filenames_once("test.tfrecords"))    

     test_label, test_features = read_and_decode(filename_queue)
     #随机散列batch,batch_size是TF每批处理的数据量
     test_batch_labels, test_batch_features = tf.test.shuffle_batch(
                                       [test_label, test_features], batch_size=batch_size)

#运行主函数
if __name__ == "__main__":
    main()


