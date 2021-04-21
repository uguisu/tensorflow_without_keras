# coding=utf-8
# author uguisu
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError


def matrix_001():

    print('\n- 矩阵转置 ------------------------')
    # 矩阵转置
    np_array = np.array([[1., 2., 3.], [4., 5., 6.]])
    tensor_a = tf.constant(np_array, dtype=tf.float32)
    tensor_b = tf.transpose(tensor_a)
    print('tensor_a shape = ', tensor_a.shape, '\n')
    print('tensor_b shape = ', tensor_b.shape, '\n')

    print('\n- 维度压缩1 ------------------------')
    # 维度压缩1
    # 删除所有维度=1的纬度
    tensor_a = tf.constant([1, 3, 4, 5], shape=(1, 4))
    print('tensor_a = ', tensor_a, '\n')
    print('tensor_a shape = ', tensor_a.shape, '\n')

    tensor_b = tf.squeeze(tensor_a)
    print('tensor_b = ', tensor_b, '\n')
    print('tensor_b shape = ', tensor_b.shape, '\n')

    print('\n- 维度压缩2 ------------------------')
    # 维度压缩2
    # input value is as follow
    # [
    #     [[1 3]]
    #     [[4 5]]
    #     [[4 6]]
    # ]
    tensor_a = tf.constant(value=[1, 3, 4, 5, 4, 6], shape=(3, 1, 2))
    print('tensor_a = ', tensor_a, '\n')
    print('tensor_a shape = ', tensor_a.shape, '\n')

    # output will be
    # [[1 3]
    #  [4 5]
    # [4
    # 6]]
    tensor_b = tf.squeeze(tensor_a)
    print('tensor_b = ', tensor_b, '\n')
    print('tensor_b shape = ', tensor_b.shape, '\n')

    print('\n- range ------------------------')
    # tf.range()用法和python的range()函数相同，不同的地方在于，循环变量(本例中的`i`)是一个tensor对象
    for i in tf.range(1, 5):
        print('i = ', i)

    print('\n- case ------------------------')
    # tf.case()用法类似于对python的if语句进行封装
    # Example 1:
    # if (x < y) return 17;
    # else return 23;
    x = 10
    y = 5
    f1 = lambda: tf.constant(17)
    f2 = lambda: tf.constant(23)
    tensor_a = tf.case([(tf.less(x, y), f1)], default=f2)
    print('tensor_a 1 = ', tensor_a, '\n')
    x = 5
    y = 10
    tensor_a = tf.case([(tf.less(x, y), f1)], default=f2)
    print('tensor_a 2 = ', tensor_a, '\n')

    # Example 2
    # if (x < y & & x > z) raise OpError("Only one predicate may evaluate to True");
    # if (x < y) return 17;
    # else if (x > z) return 23;
    # else return -1;
    def f1(): return tf.constant(17)
    def f2(): return tf.constant(23)
    def f3(): return tf.constant(-1)
    # raise InvalidArgumentError
    x = 5
    y = 10
    z = 1
    try:
        tensor_a = tf.case([(tf.less(x, y), f1), (tf.greater(x, z), f2)], default=f3, exclusive=True)
        print('tensor_a 3 = ', tensor_a, '\n')
    except InvalidArgumentError:
        print('（⊙ｏ⊙） catch InvalidArgumentError error', '\n')

    x = 5
    y = 10
    z = 20
    tensor_a = tf.case([(tf.less(x, y), f1), (tf.greater(x, z), f2)], default=f3, exclusive=True)
    print('tensor_a 4 = ', tensor_a, '\n')


if __name__ == '__main__':
    matrix_001()
