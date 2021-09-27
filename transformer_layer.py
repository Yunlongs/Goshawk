import tensorflow as tf
import tensorflow.keras.layers as layers

import time
import numpy as np


# 将位置编码矢量添加得到词嵌入，相同位置的词嵌入将会更接近，但并不能直接编码相对位置
def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# 为了避免输入中padding的token对句子语义的影响，需要将padding位mark掉，原来为0的padding项的mark输出为1
def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size,1,1,seq_len)


def compute_output_mask(seq):
    """
    因为输出的结果中包含了mask词的embedding，所以需要将这些mask词的embedding清0
    :param seq: shape=[batch_size,seq_len]
    :return:
    """
    mask = 1. - tf.cast(tf.math.equal(seq, 0), tf.float32)  # [batch_size,seq_len]
    mask = tf.expand_dims(mask, axis=2)
    real_seq_len = tf.reduce_sum(mask, axis=1)  # [batch_size,1]
    return mask, real_seq_len


def get_mean_pool(seq, out):
    """
    在输出层加一个池化，对未填充序列的embedding做mean
    :param seq: input [batch_size,seq_len]
    :param out: encoder output [batch_size,seq_len,embedding_size]
    :return:
    """
    mask, real_seq_len = compute_output_mask(seq)
    out = mask * out
    mean_pool = tf.reduce_sum(out, axis=1) / real_seq_len
    return mean_pool


# look-ahead mask 用于对未预测的token进行掩码 这意味着要预测第三个单词，只会使用第一个和第二个单词。 要预测第四个单词，仅使用第一个，第二个和第三个单词，依此类推。
def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)


# 进行attention计算的时候有3个输入 Q (query), K (key), V (value)。
# 点积注意力通过深度d_k的平方根进行缩放,因为较大的深度会使点积变大，由于使用softmax，会使梯度变小。 例如，考虑Q和K的均值为0且方差为1.它们的矩阵乘法的均值为0，方差为dk。我们使用dk的根用于缩放（而不是任何其他数字），因为Q和K的matmul应该具有0的均值和1的方差。
# 在这里我们将被掩码的token乘以-1e9(表示负无穷),这样softmax之后就为0,不对其他token产生影响。
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = layers.Softmax(axis=-1)(scaled_attention_logits)

    # attention 乘上value
    output = tf.matmul(attention_weights, v)  # （.., seq_len_v, depth）

    return output, attention_weights


# 构造mutil head attention层
class MutilHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, final_size):
        super(MutilHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.final_size = final_size

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(final_size)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        :param v:
        :param k:
        :param q: shape= [batch_size,seq_len,embedding_size]
        :param mask:
        :return:
        """
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        # 合并多头
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights


# 全连接网络
def point_wise_feed_forward_network(diff, final_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation='relu'),
        tf.keras.layers.Dense(final_size)
    ])


# 每个子层中都有残差连接，并最后通过一个正则化层。残差连接有助于避免深度网络中的梯度消失问题。
# 每个子层输出是LayerNorm(x + Sublayer(x))，规范化是在d_model维的向量上。Transformer一共有n个编码层
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ddf, final_size, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MutilHeadAttention(d_model, n_heads, final_size)
        self.ffn = point_wise_feed_forward_network(ddf, final_size)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        # 多头注意力网络
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        # assert tf.TensorShape(inputs).as_list() == tf.TensorShape(att_output).as_list(),"input's shape should equal to attention output's shape！"
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class Encoder(layers.Layer):
    def __init__(self, n_layers, d_model, embedding_size, embedding_matrix, n_heads, ddf,
                 input_vocab_size, max_seq_len, drop_rate=0.1):
        """
        tensorflow implemented transformer encoder layer.You can change your input and output size.
        :param n_layers:  how many encoder do you want to stack.
        :param d_model:   WQ,QV,WK's dims.
        :param embedding_size:  your input word's embedding size
        :param embedding_matrix:  word embedding matrix to initial.
        :param n_heads:  multihead attention.
        :param ddf:  Forward dense network (first layer)'s size.
        :param input_vocab_size:
        :param max_seq_len:
        :param drop_rate:
        """
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding_size = embedding_size
        self.final_size = embedding_size

        self.embedding = layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_size,
                                          mask_zero=True, weights=[embedding_matrix], trainable=False)
        self.pos_embedding = positional_encoding(max_seq_len, embedding_size)

        self.encode_layer = [EncoderLayer(d_model, n_heads, ddf, self.final_size, drop_rate)
                             for _ in range(n_layers)]
        self.pool_dense = layers.Dense(self.final_size, activation=tf.tanh)
        self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training, mask):
        seq_len = inputs.shape[1]
        word_emb = self.embedding(inputs)
        word_emb *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        emb = word_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.n_layers):
            x = self.encode_layer[i](x, training, mask)
        # x: shape:[batch_size,seq_len,embedding_size]
        x = self.pool_dense(x)
        return x


# 带自定义学习率调整的Adam优化器
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=40000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == "__main__":
    # create your test script here
    test_seq = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]], dtype="float")
    # mask = create_padding_mark(test_seq)
    out = np.array([[[1, 1, 1, ], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]]])
    pool = get_mean_pool(test_seq, out)
    print(pool)
