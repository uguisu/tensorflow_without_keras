# coding=utf-8
# author uguisu
import numpy as np
import tensorflow as tf

row_dim = 3
col_dim = 2
const_value = 11


def demo_zero():
    # 全零矩阵
    # def zeros(shape: {_shape_tuple},
    #           dtype: DType = dtypes.float32,
    #           name: Any = None) -> Union[object, Tensor]
    zero_tensor = tf.zeros([row_dim, col_dim])
    print('zero_tensor = ', zero_tensor, '\n')

    # 按照模板创建全零矩阵
    # 模板可以是: 数组, numpy数组, tf.constant对象
    # def zeros_like(input: {shape, dtype},
    #                   dtype: Any = None,
    #                   name: Any = None) -> Union[object, Tensor]
    numpy_template = np.array([[1, 2, 3], [11, 22, 33]])
    zero_clone_shape = tf.zeros_like(numpy_template, dtype=tf.float32)
    print('zero_clone_shape = ', zero_clone_shape, '\n')
    # shape保持不变
    assert numpy_template.shape == zero_clone_shape.shape


def demo_one():
    # 全1矩阵
    # def ones(shape: {_shape_tuple},
    #          dtype: DType = dtypes.float32,
    #          name: Any = None) -> Union[object, Tensor]
    one_tensor = tf.ones([row_dim, col_dim])
    print('one_tensor = ', one_tensor, '\n')

    # 按照模板创建全1矩阵，与`zeros_like()`函数类似
    const_template = tf.constant([[21], [22], [33]], dtype=tf.float32)
    one_clone_shape = tf.ones_like(const_template, dtype=tf.int16)
    print('one_clone_shape = ', one_clone_shape, '\n')
    # shape保持不变
    assert const_template.shape == one_clone_shape.shape
    # dtype被重新设置
    assert const_template.dtype != one_clone_shape.dtype


def demo_fill():
    # 填充矩阵
    # tf.fill(
    #     dims, value, name=None
    # )
    filled_tensor = tf.fill([row_dim, col_dim], const_value)
    print('filled_tensor = ', filled_tensor, '\n')


def demo_eye():
    # 单位矩阵
    # def eye(num_rows: Any,
    #         num_columns: Any = None,
    #         batch_shape: Any = None,
    #         dtype: DType = dtypes.float32,
    #         name: Any = None) -> object
    eye_001 = tf.eye(3, 3, dtype=tf.int32)
    print('eye_001 = ', eye_001, '\n')


def demo_linspace():
    """
    线性张量
    """

    # 生成一组线性张量，类似于range()
    # 范围: [start, stop]
    # tf.linspace(
    #     start, stop, num, name=None, axis=0
    # )
    start_num = 0.0
    stop_num = 6.0
    sequence_increase = 2
    num = int((stop_num - start_num) / sequence_increase + 1)
    lin_space_tensor = tf.linspace(start=start_num, stop=stop_num, num=num)
    print('lin_space_tensor = ', lin_space_tensor, '\n')

    # 范围: [start, stop)
    buildin_range = [x for x in range(int(start_num), int(stop_num), sequence_increase)]
    print('buildin_range = ', buildin_range)


def confirm_version():
    print(tf.__version__)


if __name__ == '__main__':
    confirm_version()
    demo_zero()
    demo_one()
    demo_fill()
    demo_eye()
    demo_linspace()
