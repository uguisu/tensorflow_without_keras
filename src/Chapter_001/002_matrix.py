# coding=utf-8
# author uguisu
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.stats import gamma


def matrix():
    """
    矩阵运算
    """

    # 单位矩阵
    # def eye(num_rows: Any,
    #         num_columns: Any = None,
    #         batch_shape: Any = None,
    #         dtype: DType = dtypes.float32,
    #         name: Any = None) -> object
    mt_001 = tf.eye(3, 3, dtype=tf.int32)
    print('mt_001 = ', mt_001, '\n')

    # 全零矩阵
    # def zeros(shape: {_shape_tuple},
    #           dtype: DType = dtypes.float32,
    #           name: Any = None) -> Union[object, Tensor]
    mt_002 = tf.zeros((3, 3), dtype=tf.int32)
    print('mt_002 = ', mt_002, '\n')

    # 按照模板创建全零矩阵
    # 模板可以是: 数组, numpy数组, tf.constant对象
    # def zeros_like(input: {shape, dtype},
    #                   dtype: Any = None,
    #                   name: Any = None) -> Union[object, Tensor]
    numpy_template = np.array([[1, 2, 3], [11, 22, 33]])
    mt_002_clone_shape = tf.zeros_like(numpy_template, dtype=tf.float32)
    print('mt_002_clone_shape = ', mt_002_clone_shape, '\n')
    # shape保持不变
    assert numpy_template.shape == mt_002_clone_shape.shape

    # 全1矩阵
    # def ones(shape: {_shape_tuple},
    #          dtype: DType = dtypes.float32,
    #          name: Any = None) -> Union[object, Tensor]
    mt_003 = tf.ones((3, 3), dtype=tf.int32)
    print('mt_003 = ', mt_003, '\n')

    # 按照模板创建全1矩阵，与`zeros_like()`函数类似
    const_template = tf.constant([[21], [22], [33]], dtype=tf.float32)
    mt_003_clone_shape = tf.zeros_like(const_template, dtype=tf.int16)
    print('mt_003_clone_shape = ', mt_003_clone_shape, '\n')
    # shape保持不变
    assert const_template.shape == mt_003_clone_shape.shape
    # dtype被重新设置
    assert const_template.dtype != mt_003_clone_shape.dtype


def random_uniform():
    """
    从"服从指定均匀分布的序列"中随机取出指定个数的值。
    - `均匀分布`是指，二维空间中, x = minval and x = maxval 的一条直线
    """

    # 函数原型
    # def random_uniform(shape: Any,
    #                    minval: int = 0,
    #                    maxval: Optional[{__eq__}] = None,
    #                    dtype: DType = dtypes.float32,
    #                    seed: Any = None,
    #                    name: Any = None) -> object
    # shape – A 1-D integer Tensor or Python array. The shape of the output tensor.
    rnd = tf.random.uniform([1, 2], minval=1, maxval=5, dtype=tf.float32)
    print('uniform = ', rnd, '\n')


def random_normal():
    """
    从"服从指定正态分布的序列"中随机取出指定个数的值。
    """

    # 函数原型
    # def random_normal(shape: Any,
    #                   mean: float = 0.0,
    #                   stddev: float = 1.0,
    #                   dtype: DType = dtypes.float32,
    #                   seed: Any = None,
    #                   name: Any = None) -> object
    # shape – A 1-D integer Tensor or Python array. The shape of the output tensor.
    rnd = tf.random.normal([12], mean=0, stddev=1, dtype=tf.float32)
    print('normal 1 = ', rnd, '\n')
    print('normal 1 shape = ', rnd.shape, '\n')

    # 注意维度的不同
    rnd = tf.random.normal([12, 1], mean=0, stddev=1, dtype=tf.float32)
    print('normal 2 = ', rnd, '\n')
    print('normal 2 shape = ', rnd.shape, '\n')


def random_poisson():
    """
    从"服从指定泊松分布的序列"中随机取出指定个数的值。
    """

    def _draw():
        # Poisson分布
        x = np.random.poisson(lam=1.5, size=10000)
        pillar = 15
        a = plt.hist(x, bins=pillar, range=[0, pillar], color='g', alpha=0.5)
        plt.plot(a[1][0:pillar], a[0], 'r')
        plt.grid()
        plt.show()

    # 函数原型
    # def random_poisson_v2(shape: Any,
    #                       lam: Any,
    #                       dtype: DType = dtypes.float32,
    #                       seed: Any = None,
    #                       name: Any = None) -> Any
    # shape – A 1-D integer Tensor or Python array.
    #         The shape of the output samples to be drawn per "rate"-parameterized distribution.
    #         注意: 这里的shape还有一个隐藏的纬度，这个纬度的大小是根据参数`lam`的维度来确定的
    # lam - A Tensor or Python value or N-D array of type dtype.
    #       lam provides the rate parameter(s) describing the poisson distribution(s) to sample.
    #       对于lam的理解，就是设置一组λ(例如 [0.5, 1.5]), 然后生成满足泊松分布的随机数，有几个λ就生成几组。生成的结果与shape的最后一维
    #       融合。
    rnd = tf.random.poisson([10], [0.5, 1.5], dtype=tf.float32)
    print('poisson1 shape = ', rnd.shape, '\n')
    rnd = tf.random.poisson([5, 7], [0.5, 1.5, 5.5], dtype=tf.float32)
    print('poisson2 = ', rnd, '\n')
    print('poisson2 shape = ', rnd.shape, '\n')

    _draw()


def random_categorical():
    """
    分类采样
    """

    # 函数原型
    # def categorical(logits: Any,
    #                 num_samples: Any,
    #                 dtype: Any = None,
    #                 seed: Any = None,
    #                 name: Any = None) -> object
    # logits - 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized
    #          log-probabilities for all classes.
    # num_samples - 0-D. Number of independent samples to draw for each row slice.
    # Return - The drawn samples of shape [batch_size, num_samples].
    # 理解：给一个[batch_size * num_classes]的方阵，`batch_size`表示采样次数，`num_classes`表示每一个类别在每一个`batch`中被抽取的概率
    #      `num_samples`表示每一个`batch`抽取样本的数量。
    #      返回值是[batch_size * num_samples]的方阵，里面的数字不是样本本身，类别的下标

    num_samples = 5
    # 均匀抽取
    logits = tf.math.log([[0.2, 0.2, 0.2, 0.2, 0.2],
                         [1., 1., 1., 1., 1.]])
    cat = tf.random.categorical(logits, num_samples)
    print('logits shape = ', logits.shape, '\n')
    print('cat1 = ', cat, '\n')

    # batch 0 倾向于样本4, batch 1 倾向于样本0
    logits = tf.math.log([[0.1, 0.1, 0.1, 0.1, 1.],
                         [1., 0.1, 0.1, 0.1, 0.1]])
    cat = tf.random.categorical(logits, num_samples)
    print('cat2 = ', cat, '\n')


def random_gamma():
    """
    从"服从指定gamma分布的序列"中随机取出指定个数的值。
    """

    def _draw():
        # Gamma分布（Gamma Distribution）
        alpha_values = [1, 2, 3, 3, 3]
        beta_values = [0.5, 0.5, 0.5, 1, 2]
        color = ['b', 'r', 'g', 'y', 'm']
        x = np.linspace(1E-6, 10, 1000)

        fig, ax = plt.subplots(figsize=(12, 8))

        for k, t, c in zip(alpha_values, beta_values, color):
            dist = gamma(k, 0, t)
            plt.plot(x, dist.pdf(x), c=c, label=r'$\alpha=%.1f,\beta=%.1f$' % (k, t))

        plt.xlim(0, 10)
        plt.ylim(0, 2)

        plt.xlabel('$x$')
        plt.ylabel(r'$p(x|\alpha,\beta)$')
        plt.title('Gamma Distribution')

        plt.legend(loc=0)
        plt.show()

    # 函数原型
    # def random_gamma(shape: Any,
    #                  alpha: Any,
    #                  beta: Any = None,
    #                  dtype: DType = dtypes.float32,
    #                  seed: Any = None,
    #                  name: Any = None) -> Any
    # shape - A 1-D integer Tensor or Python array. The shape of the output samples to be drawn per
    #         alpha/beta-parameterized distribution.
    #         注意: 这里的shape还有两个隐藏的纬度，这个纬度的大小是根据参数`alpha`和`beta`的维度来确定的
    # alpha - A Tensor or Python value or N-D array of type dtype. alpha provides the shape parameter(s) describing
    #         the gamma distribution(s) to sample. Must be broadcastable with beta.
    # beta - A Tensor or Python value or N-D array of type dtype. Defaults to 1. beta provides the inverse scale
    #        parameter(s) of the gamma distribution(s) to sample. Must be broadcastable with alpha.
    gm = tf.random.gamma([5], [0.5, 1.5])
    print('gm1 = ', gm, '\n')
    print('gm1 shape = ', gm.shape, '\n')

    # alpha + beta
    alpha = tf.constant([[1.], [3.], [5.]])
    beta = tf.constant([[3., 4.]])
    gm = tf.random.gamma([5], alpha=alpha, beta=beta)
    # samples has shape [5, 3, 2], with 5 samples each of 3x2 distributions.
    print('gm2 = ', gm, '\n')

    _draw()


def linear_algebra():
    """
    线性代数 Y = XW + b
    """

    # 样本数量
    n = 100

    # 生成测试用数据集
    X = tf.random.uniform([n, 2], minval=-10, maxval=10)
    w0 = tf.constant([[1.0], [2.0]])
    b0 = tf.constant(3.0)

    # 矩阵乘法,增加正态扰动
    Y = tf.matmul(X, w0) + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0, dtype=tf.float32)

    # 初始化权重为正态分布
    w = tf.Variable(tf.random.normal(w0.shape, mean=0.0, stddev=1.0, dtype=tf.float32))
    # 初始化噪声=0
    b = tf.Variable(0.0)
    # 训练次数
    epoches = 5000
    # learn rate
    learn_rate = 0.001

    for epoch in tf.range(1, epoches + 1):
        # eager model
        with tf.GradientTape() as tape:
            # 正向传播求损失
            Y_hat = tf.matmul(X, w) + b
            loss = tf.squeeze(tf.matmul(tf.transpose(Y - Y_hat), (Y - Y_hat))) / (2.0 * n)

        # 反向传播求梯度
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
        del tape

        # 梯度下降法更新参数
        w.assign_sub(learn_rate * dloss_dw)
        b.assign_sub(learn_rate * dloss_db)

        if epoch % 1000 == 0:
            tf.print("epoch =", epoch, " loss =", loss, )
            tf.print("w =", w)
            tf.print("b =", b)
            tf.print("")


if __name__ == '__main__':
    matrix()
    random_uniform()
    random_normal()
    random_poisson()
    random_categorical()
    random_gamma()
    linear_algebra()
