# coding=utf-8
# author uguisu
import numpy as np
import tensorflow as tf


def constant_basic():
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
    print('const_001 = ', const_001, '\n')

    # 将numpy数组转换成常量
    numpy_const_001 = np.array(np.array([[1, 2, 3], [4, 5, 6]]))
    const_002 = tf.constant(numpy_const_001)
    print('const_002 = ', const_002, '\n')

    # `dtype`属性会根据数据的类型自动确定
    print('const_001 dtype = ', const_001.dtype, '\n')

    # 如果手工指定了`dtype`，则数据会自动进行转换
    const_001_float32 = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    print('const_001_float32 = ', const_001_float32, '\n')
    print('const_001_float32 dtype = ', const_001_float32.dtype, '\n')
    # 对比`const_001`的输出，不难看出，数据类型已经发生了改变

    # 如果在创建常量的同时，给定了形状`shape`，tf会自动尝试按照指定的形状进行填充
    const_003_with_shape = tf.constant(np.array([1, 2, 3, 4, 5, 6]), shape=(3, 2))
    print('const_003_with_shape = ', const_003_with_shape, '\n')
    # 即使传入数据的形状与`shape`参数不相同，数据也会按照`shape`进行重新填充
    const_004_with_shape = tf.constant(np.array(np.array([[1, 2, 3], [4, 5, 6]])), shape=(3, 2))
    print('const_004_with_shape = ', const_004_with_shape, '\n')


def constant_calculate():
    """
    常量计算
    """

    print('\n\n- 常数计算 ------------------------')
    # 常数计算
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = x + 2
    print('y1 = ', y, '\n')
    y = x * 3
    print('y2 = ', y, '\n')
    y = x ** 2
    print('y3 = ', y, '\n')
    y = x / 3
    print('y4 = ', y, '\n')

    # 注意：若定义`x = tf.constant([[1, 2], [3, 4]], dtype=tf.int8)`，则会报错
    # tensorflow.python.framework.errors_impl.InvalidArgumentError: Value
    # for attr 'T' of int8 is not in the list of allowed values: bfloat16, float, half, double, int32, int64, complex64,
    # complex128; NodeDef: {{node Pow}}; Op < name = Pow;
    # signature = x:T, y: T -> z: T;
    # attr = T:type, allowed = [DT_BFLOAT16, DT_FLOAT, DT_HALF, DT_DOUBLE, DT_INT32, DT_INT64, DT_COMPLEX64,
    #                           DT_COMPLEX128] > [Op:Pow]

    print('\n\n- 两个常量计算 ------------------------')
    # 两个常量计算
    x1 = tf.constant([[11, 12], [21, 22]], dtype=tf.float32)
    x2 = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    y = x1 + x2
    print('y5 = ', y, '\n')
    y = x1 - x2
    print('y6 = ', y, '\n')
    y = x1 * x2
    print('y7 = ', y, '\n')
    y = x1 / x2
    print('y8 = ', y, '\n')

    y = pow(x1, 1 / 2)
    print('y9 = ', y, '\n')


def calculate():

    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = tf.constant([1, 0], dtype=tf.float32, shape=(2, 1))
    print(tf.matmul(x, y), '\n')
    print(x @ y, '\n')


if __name__ == '__main__':
    constant_basic()
    constant_calculate()
    calculate()
