# coding=utf-8
# author uguisu
import tensorflow as tf


def demo_variable_declare():
    # 定义变量
    # tf.Variable(
    #     initial_value=None, trainable=None, validate_shape=True, caching_device=None,
    #     name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
    #     synchronization=tf.VariableSynchronization.AUTO,
    #     aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
    # )
    #
    # TODO 参数翻译可能不准确
    # initial_value - 初始值，可以为数字或者可转化成Tensor的python对象
    # trainable - 标价此变量是否参与训练。
    #     True(默认值): GradientTapes会自动监视此变量
    #     False: 当synchronizatio被设置为ON_READ时，此参数是False
    # validate_shape - 是否已知变量的shape
    #     True(默认值): initial_value所对应的值必须有明确的shape
    #     False: 允许initial_value所对应的值的shape未知
    # caching_device - 指定一个外部缓存设备. 仅在使用tf v1 风格的Sessio时才有效.
    # variable_def - VariableDef协议缓冲区. 如果不是 None, 则通过引用计算图中变量的节点, 重新创建 Variable 对象及其内容,
    #     这个被引用的节点必须存在. 计算图本身并不会被改变.
    #     注意: variable_def和其他参数是互斥的
    # synchronization - tf.VariableSynchronization中定义的某一种类型. 默认值是AUTO

    v = tf.Variable(1.0)

    print('v = ', v)
    print('v.value = ', v.value())
    assert 1.0 == v.value()


def demo_variable_calculate():
    # 初始化一个变量
    v_2 = tf.Variable(2.0)
    print('v_2.value = ', v_2.value())
    assert 2.0 == v_2.value()

    # 不能使用等号来给变量赋值，需要使用函数assign()
    v_2.assign(2.1)
    print('new v_2.value = ', v_2.value())
    assert 2.1 == v_2.value()

    # 变量不能直接, 和数字相加, 下面的操作会抛出AttributeError
    # try:
    #     v_2 = v_2 + 3
    # except AttributeError:
    #     print('wow, 不能直接与数字相加')
    #
    # 注意!!! 这个try方法一旦执行, v_2 的类型就会发生改变, 并导致assign_add()发生错误
    # AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'assign_add'

    # 要和数字相加, 需要使用函数assign_add()
    v_2.assign_add(0.4)
    print('new2 v_2.value = ', v_2.value())
    assert 2.5 == v_2.value()

    # 乘法
    # w: [ 1,
    #      2]
    # x = [3, 4 ]
    w = tf.Variable([[1.], [2.]])
    x = tf.constant([[3., 4.]])
    v_3 = tf.matmul(w, x)
    # 注意!!! v_3 也会变成 EagerTensor 类型，所以不能直接使用value()方法
    # print('v_3.value = ', v_3.value())
    print('v_3 = ', v_3)


def confirm_version():
    print('tensorflow version = ', tf.__version__)
    print('eager = ', tf.executing_eagerly())


if __name__ == '__main__':
    confirm_version()
    demo_variable_declare()
    demo_variable_calculate()
