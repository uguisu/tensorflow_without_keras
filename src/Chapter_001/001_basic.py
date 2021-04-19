# coding=utf-8
# author uguisu
import numpy as np
import tensorflow as tf


def my_constant():
    """
    常量
    """

    # 函数原型
    # https://tensorflow.google.cn/versions/r2.3/api_docs/python/tf/constant?hl=en
    # tf.constant(
    #     value, dtype=None, shape=None, name='Const'
    # )

    # 直接定义常量
    const_001 = tf.constant([1, 2, 3, 4, 5, 6])
    print('const_001 = ', const_001)

    # 将numpy数组转换成常量
    numpy_const_001 = np.array(np.array([[1, 2, 3], [4, 5, 6]]))
    const_002 = tf.constant(numpy_const_001)
    print('const_002 = ', const_002)

    # `dtype`属性会根据数据的类型自动确定
    print('const_001 dtype = ', const_001.dtype)

    # 如果手工指定了`dtype`，则数据会自动进行转换
    const_001_float32 = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    print('const_001_float32 = ', const_001_float32)
    print('const_001_float32 dtype = ', const_001_float32.dtype)
    # 对比`const_001`的输出，不难看出，数据类型已经发生了改变

    # 如果在创建常量的同时，给定了形状`shape`，tf会自动尝试按照指定的形状进行填充
    const_003_with_shape = tf.constant(np.array([1, 2, 3, 4, 5, 6]), shape=(3, 2))
    print('const_003_with_shape = ', const_003_with_shape)
    # 即使传入数据的形状与`shape`参数不相同，数据也会按照`shape`进行重新填充
    const_004_with_shape = tf.constant(np.array(np.array([[1, 2, 3], [4, 5, 6]])), shape=(3, 2))
    print('const_004_with_shape = ', const_004_with_shape)


if __name__ == '__main__':

    my_constant()
