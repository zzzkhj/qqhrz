import math
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


def mask_seq(seqs: Tensor, vaild_lens: Tensor, mask_value=0.0) -> Tensor:
    """
    对序列的无效的元素用mask_value替换
    :param seqs: (n, m)
    :param vaild_lens: (n*m, )
    :param mask_value: 数值
    :return: (n, m)
    """
    # seqs: (batch_size, num_steps)
    # valid_lens: (batch_size, )
    # mask: seqs.shape, 每一行对应的是不要屏蔽的元素就是1, 要屏蔽的就是0
    max_len = seqs.shape[1]
    mask = torch.arange(max_len, dtype=torch.float32, device=seqs.device)[None, :] >= vaild_lens[:, None]
    # 把零元素都换成屏蔽元素的值
    seqs[mask] = mask_value
    return seqs


class MaskSoftmask(nn.Module):
    """
    掩蔽softmax
    """

    def __init__(self, dim: int):
        """
        :param dim: 进行softmax的维度
        """
        super(MaskSoftmask, self).__init__()
        self.dim = dim

    def forward(self, X: Tensor, vaild_lens=None) -> Tensor:
        """
        :param X: (batch_size, num_queries, num_keys/num_values)
        :param vaild_lens: (batch_size, ), 每个value的有效长度
        :return: (batch_size, num_queries, num_keys/num_values)
        """
        # X: (batch_size, num_query, num_keys)
        # valid_lens: (batch_size, )
        if vaild_lens is None:
            return F.softmax(X, dim=self.dim)
        shape = X.shape
        if vaild_lens.dim() == 1:
            vaild_lens = vaild_lens.repeat_interleave(shape[1])
        else:
            vaild_lens = vaild_lens.reshape(-1)  # 变成一维
        mask_X = mask_seq(X.reshape(-1, shape[-1]), vaild_lens, -1e6).reshape(shape)
        return F.softmax(mask_X, dim=self.dim)


class MaskCrossEntropyLoss(nn.Module):
    """隐蔽CrossEntropyLoss"""

    def __init__(self):
        super(MaskCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, label, vaild_lens):
        # pred: (batch_size, num_steps, vocab_size)
        # label: (batch_size, num_steps)
        weight = torch.ones_like(label)
        weight = mask_seq(weight, vaild_lens)
        unweight_loss = self.loss_function(pred.reshape(-1, pred.shape[-1]), label.reshape(-1)).reshape(label.shape)
        weight_loss = (unweight_loss * weight).sum(dim=1)
        return weight_loss


class AdditiveAtttention(nn.Module):
    """加性注意力"""

    def __init__(self, query_features: int, key_features: int, num_hiddens: int, dropout=0.1):
        """
        :param query_features: query的特征维度
        :param key_features: key的特征维度
        :param num_hiddens: 隐层特征维度
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super().__init__()
        self.attention_weights = None
        # 将queries和keys的特征维度统一
        self.dense_query = nn.Linear(query_features, num_hiddens)
        self.dense_key = nn.Linear(key_features, num_hiddens)
        # 相加后将num_hiddens映射到1个特征，因为每个query对每个value只有一个权重
        self.dense_query_add_key = nn.Linear(num_hiddens, 1)
        self.mask_softmax = MaskSoftmask(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor) -> Tensor:
        """
        前向传播
        :param queries: (batch_size, num_queries, query_features)
        :param keys: (batch_size, num_keys, key_features)
        :param values: (batch_size, num_values, value_features)
        :param valid_lens: (batch_size, )
        :return: 所有value的加权和: (batch_size, num_queries, value_features)
        """
        queries, keys = self.dense_query(queries), self.dense_key(keys)
        q_k = queries.unsqueeze(2) + keys.unsqueeze(1)  # 广播机制，使得可以相加
        scores = torch.tanh(self.dense_query_add_key(q_k).squeeze(-1))
        self.attention_weights = self.mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力，要求query和key有同样的特征维度"""

    def __init__(self, dropout=0.1):
        """
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super(DotProductAttention, self).__init__()
        self.attention_weights = None
        self.mask_softmax = MaskSoftmask(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor) -> Tensor:
        """
        要求query和key有同样的特征维度, 即query_features=key_features
        :param queries: (batch_size, num_queries, query_features)
        :param keys: (batch_size, num_keys, key_features)
        :param values: (batch_size, num_values, value_features)
        :param valid_lens: (batch_size, )
        :return: 所有value的加权和: (batch_size, num_queries, value_features)
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # keys.permute(0, 2, 1)
        self.attention_weights = self.mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, query_features: int, key_features: int, value_features: int, num_hiddens: int, num_heads: int,
                 dropout, use_bias=False):
        """
        :param query_features: query的特征维度
        :param key_features: key的特征维度
        :param num_hiddens: 隐层特征维度
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super().__init__()
        self.attentiion_weights = None
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        # 将queries、keys和values的特征维度统一
        self.dense_query = nn.Linear(query_features, num_hiddens, use_bias)
        self.dense_key = nn.Linear(key_features, num_hiddens, use_bias)
        self.dense_value = nn.Linear(value_features, num_hiddens, use_bias)
        # 注意力
        self.attention = DotProductAttention(dropout)
        self.mask_softmax = MaskSoftmask(dim=-1)
        self.dense_out = nn.Linear(num_hiddens, num_hiddens, use_bias)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens=None) -> Tensor:
        """
        前向传播
        :param queries: (batch_size, num_queries, query_features)
        :param keys: (batch_size, num_keys, key_features)
        :param values: (batch_size, num_values, value_features)
        :param valid_lens: (batch_size, )
        :return: 所有value的加权和: (batch_size, num_queries, value_features)
        """
        # 下面的操作就是把多头做并行计算，并且改变形状，方便后续恢复
        # quries: (batch_size * num_heads, num_queries, num_hiddens // num_heads)
        # keys: (batch_size * num_heads, num_keys, num_hiddens // num_heads)
        # values: (batch_size * num_heads, num_values, num_hiddens // num_heads)
        # 相当于有batch_size * num_heads个样本，且每batch_size个样本都属于同一个头
        shape = queries.shape[:2] + (self.num_hiddens,)
        queries, keys, values = self.dense_query(queries).reshape(-1, queries.shape[1],
                                                                  self.num_hiddens // self.num_heads), \
                                self.dense_key(keys).reshape(-1, keys.shape[1], self.num_hiddens // self.num_heads), \
                                self.dense_value(values).reshape(-1, values.shape[1],
                                                                 self.num_hiddens // self.num_heads)
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(self.num_heads, dim=0)
        # output_concat，形状：(num_heads * batch_size, num_queries, num_hiddens // num_heads)
        output_concat = self.attention(queries, keys, values, valid_lens)
        # 恢复形状: (batch_size, num_queries, num_hiddens)
        output = self.dense_out(output_concat.reshape(shape))
        return output


class PositionEncoder(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens: int, dropout: float, max_len=1000):
        """
        位置编码
        :param num_hiddens: X的特征维度
        :param dropout: 丢弃百分之dropout，防止过拟合
        :param max_len: 序列最大长度
        """
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((max_len, num_hiddens))
        # pv: (max_len, num_hiddens / 2 向上取整)
        pv = torch.arange(max_len).reshape(-1, 1) / torch.pow(10000,
                                                              torch.arange(0, num_hiddens, 2) / num_hiddens)  # 广播机制
        self.P[:, 0::2] = torch.sin(pv)  # 偶数列使用sin
        # num_hiddens为奇数时，pv多了最后一列
        i = 0 if pv.shape[1] % 2 == 0 else 1
        self.P[:, 1::2] = torch.cos(pv[:, :pv.shape[1] - i])  # 奇数列使用cos

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (batch_size, num_steps, num_hiddens)
        :return: (batch_size, num_steps, num_hiddens)
        """
        X = X + self.P[:X.shape[1], :].to(X.device)  # 广播机制
        return self.dropout(X)


class AddNorm(nn.Module):
    """残差连接后层规范化"""

    def __init__(self, normalized_shape, dropout: float):
        """
        :param normalized_shape: 层规范化后的形状
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm(normalized_shape)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        残差连接后层规范化
        :param X:
        :param Y:
        :return: 残差连接和层规范化后的结果
        """
        return self.lay_norm(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈神经网络"""

    def __init__(self, ffn_in_features, ffn_num_hiddens, ffn_out_features):
        """
        基于位置的前馈神经网络
        :param ffn_in_features: 输入的特征数
        :param ffn_num_hiddens: 隐层特征数
        :param ffn_out_features: 输出特征数
        """
        super().__init__()
        self.dense1 = nn.Linear(ffn_in_features, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_out_features)

    def forward(self, X: Tensor) -> Tensor:
        return self.dense2(self.relu(self.dense1(X)))


class TransformerEncoderBlock(nn.Module):
    """Transformer编码块"""

    def __init__(self, query_features: int, key_features: int, value_features: int, num_hiddens: int, num_heads: int,
                 normalized_shape, ffn_num_hiddens, dropout, use_bias=False):
        """
        Transformer编码块
        :param query_features: query的特征维度
        :param key_features: key的特征维度
        :param value_features: value的特征维度
        :param num_hiddens: 隐层特征维度
        :param num_heads: 头的数量
        :param normalized_shape: 层规范化后的形状
        :param ffn_num_hiddens: 基于位置的前馈神经网络的隐层特征维度
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super().__init__()
        self.attention = MultiHeadAttention(query_features, key_features, value_features, num_hiddens, num_heads,
                                            dropout, use_bias)
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)  # 保持形状不变
        self.add_norm2 = AddNorm(normalized_shape, dropout)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        Y = self.attention(X, X, X, valid_lens)  # 多头自注意力
        Y2 = self.ffn(self.add_norm1(X, Y))
        Y3 = self.add_norm2(Y, Y2)
        return Y3


class TransformerDecoderBlock(nn.Module):
    """Transformer解码块"""

    def __init__(self, query_features: int, key_features: int, value_features: int, num_hiddens: int, num_heads: int,
                 normalized_shape, ffn_num_hiddens: int, i: int, dropout=0.1):
        """
        Transformer解码块
        :param query_features: query的特征维度
        :param key_features: key的特征维度
        :param value_features: value的特征维度
        :param num_hiddens: 隐层特征维度
        :param num_heads: 头的数量
        :param normalized_shape: 层规范化后的形状
        :param ffn_num_hiddens: 基于位置的前馈神经网络的隐层特征维度
        :param i: Transformer解码器中的第i个块
        :param dropout: 丢弃百分之dropout的数据，防止过拟合
        """
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(query_features, key_features, value_features, num_hiddens, num_heads,
                                             dropout)
        self.add_norm1 = AddNorm(normalized_shape, dropout)
        self.attention2 = MultiHeadAttention(query_features, key_features, value_features, num_hiddens, num_heads,
                                             dropout)
        self.add_norm2 = AddNorm(normalized_shape, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)  # 保持形状不变
        self.add_norm3 = AddNorm(normalized_shape, dropout)

    def forward(self, X: Tensor, state: Tensor) -> tuple:
        """
        训练时是并行计算的，预测时是根据前一个预测结果预测下一个结果（第一次是编码器的输入和初始state作为输入，后续都是前面的预测结果和state作为输入）
        :param X: (batch_size, num_steps, num_hiddens)
        :param state: 元组，其中三个元素分别是编码器的输出、编码的有效长度和Transformer解码器中每个解码块的state
        :return: 解码结果和state
        """
        # 训练时，encoder_output，encoder_valid_lens是编码器的输出
        # 预测时，encoder_output，encoder_valid_lens第一次是编码器的输入和初始state作为输入，后续都是前面的预测结果和输出state作为输入
        # 注意，预测时，state[0]和state[1]是不变的，变化的只有state[2]
        encoder_output, encoder_valid_lens = state[0], state[1]
        # 训练时：state[2][self.i]都是None，因为是并行计算
        # 测试时，state[2][self.i]除一开始都是None，后面存的都是Transformer解码器的第i解码块直到当前时间步前的所有预测结果的拼接
        if state[2][self.i] is None:
            keys = X
            values = X
        else:
            keys = torch.cat([state[2][self.i], X], dim=1)
            values = torch.cat([state[2][self.i], X], dim=1)

        if self.training:
            batch_size, num_steps = X.shape[0], X.shape[1]
            # 因为是并行的，所有让其只能看到从开始到自己位置的序列
            # decoder_valid_lens: (batch_size, num_steps)，每行都是1, 2, 3, ..., num_steps
            decoder_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            decoder_valid_lens = None
        Y = self.attention1(X, keys, values, decoder_valid_lens)  # 掩蔽多头自注意力，关注解码器的输入
        Y2 = self.add_norm1(X, Y)

        Y3 = self.attention2(Y2, encoder_output, encoder_output, encoder_valid_lens)  # 关注编码器的输出
        Y4 = self.add_norm2(Y2, Y3)

        Y5 = self.ffn(Y4)
        Y6 = self.add_norm3(Y4, Y5)
        return Y6, state


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, vocab_size, num_hiddens, num_lays, num_heads, normalized_shape, ffn_num_hiddens, dropout,
                 use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoder = PositionEncoder(num_hiddens, dropout)
        # 编码块
        self.blocks = nn.Sequential()
        for i in range(num_lays):
            self.blocks.add_module(f'block{i}', TransformerEncoderBlock(num_hiddens, num_hiddens, num_hiddens,
                                                                        num_hiddens, num_heads, normalized_shape,
                                                                        ffn_num_hiddens, dropout, use_bias))

    def forward(self, X, valid_lens):
        # self.embedding(X)的输出结果相较于位置编码（值是-1~1）较小，所有先将其进行缩放再与位置编码相加
        X2 = self.position_encoder(self.embedding(X) * math.sqrt(self.num_hiddens))
        for block in self.blocks:
            X2 = block(X2, valid_lens)
        return X2


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, vocab_size, num_hiddens, num_lays, num_heads, normalized_shape, ffn_num_hiddens, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_lays = num_lays
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_encoder = PositionEncoder(num_hiddens, dropout)
        # 解码器块
        self.blocks = nn.Sequential()
        for i in range(num_lays):
            self.blocks.add_module(f"block{i}",
                                   TransformerDecoderBlock(num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                                                           num_heads, normalized_shape, ffn_num_hiddens, i, dropout))
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state):
        X2 = self.position_encoder(self.embedding(X) * math.sqrt(self.num_hiddens))
        for block in self.blocks:
            X2, state = block(X2, state)
        return self.out(X2), state

    def init_state(self, encoder_outputs, encoder_valid_lens):
        return [encoder_outputs, encoder_valid_lens, [None] * self.num_lays]


class EncoderDecoder(nn.Module):
    """编码器解码器模型"""

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_X, decoder_X, encoder_valid_lens):
        encoder_outputs = self.encoder(encoder_X, encoder_valid_lens)
        state = self.decoder.init_state(encoder_outputs, encoder_valid_lens)
        return self.decoder(decoder_X, state)


def get_tokens_and_segments(token_a, token_b=None):
    """
    为token_a和token_b加上<cls>和<seq>，并对每个句子进行标记存在segments中，其中0代表token_a，1表示token_b。
    :param token_a: 字符串列表，每个元素是一个词
    :param token_b: 字符串列表，每个元素是一个词，默认值是None
    :return: tokens和segments
    """
    tokens = ['<cls>'] + token_a + ['<seq>']
    segments = [0] * len(tokens)
    if token_b is None:
        return tokens, segments
    else:
        tokens = tokens + token_b + ['<seq>']
        segments += [1] * (len(token_b) + 1)
        return tokens, segments


class BERTEncoder(nn.Module):
    """BERT编码器"""

    def __init__(self, vocab_size, num_hiddens, num_lays, num_heads, normalized_shape, ffn_num_hiddens, dropout,
                 max_len=1000):
        super(BERTEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.postion_embedding = nn.Parameter(torch.randn(max_len, num_hiddens))
        self.blocks = nn.Sequential()
        for i in range(num_lays):
            self.blocks.add_module(f'block{i}',
                                   TransformerEncoderBlock(num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                                                           num_heads, normalized_shape, ffn_num_hiddens, dropout))

    def forward(self, tokens, segments, valid_lens):
        X = self.embedding(tokens) + self.embedding(segments) + self.postion_embedding.data[:tokens.shape[1], :]
        for block in self.blocks:
            X = block(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """掩蔽语言模型"""

    def __init__(self, vocab_size: int, in_features: int, num_hiddens: int) -> None:
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, predict_positions):
        """
        掩蔽语言模型
        :param X: (batch_size, num_steps, embed_size)
        :param predict_positions: (batch_size, num_predictions)
        :return: (batch_size, num_predictions, vocab_size)，返回预测结果
        """
        batch_size = X.shape[0]
        num_predictions = predict_positions.shape[1]  # 每个序列中预测的词的个数
        row_indexs = torch.arange(batch_size).repeat_interleave(num_predictions)  # 预测词在X的行索引
        clomun_indexs = predict_positions.reshape(-1)  # 预测词在X的列索引
        predict_words = X[row_indexs, clomun_indexs].reshape(batch_size, num_predictions, -1)  # 通过行索引和列索引获得要预测的词的词向量
        return self.mlp(predict_words)


class NextSentencePredict(nn.Module):
    """下一句预测"""

    def __init__(self, in_features: int):
        super(NextSentencePredict, self).__init__()
        self.out = nn.Linear(in_features, 2)  # 二分类，是或不是

    def forward(self, X):
        """
        下一句预测
        :param X: (batc_size, in_features)，在bert模型中<cls>词元的向量表示整个句子的向量
        :return: 预测结果，(batc_size, 2)
        """
        return self.out(X)


class BERTModoul(nn.Module):
    """BERT模型"""

    def __init__(self, vocab_size, num_hiddens, num_lays, num_heads, normalized_shape, ffn_num_hiddens, dropout,
                 max_len=1000):
        super(BERTModoul, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, num_lays, num_heads, normalized_shape,
                                   ffn_num_hiddens, dropout, max_len)
        self.dense = nn.Sequential(nn.Linear(num_hiddens, num_hiddens),
                                   nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, num_hiddens)
        self.nsp = NextSentencePredict(num_hiddens)

    def forward(self, tokens, segments, predict_positions=None, valid_lens=None):
        X = self.encoder(tokens, segments, valid_lens)
        if predict_positions is not None:
            mlm_Y_hat = self.mlm(X, predict_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.dense(X[:, 0, :]))
        return X, mlm_Y_hat, nsp_Y_hat


def get_next_sentence(sentence, next_sentence, all_sentcences):
    """
    对每个句子生成下一句，50%的时候是真实的下一句，50%的时候随机选择一句
    :param sentence: 第一个句子
    :param next_sentence: 下一句
    :param all_sentcences: 所有句子的列表
    :return: sentence, next_sentence, all_sentcences
    """
    is_next = 1
    if random.random() < 0.5:
        next_sentence = random.choice(all_sentcences)
        is_next = 0
    return sentence, next_sentence, is_next


def get_maskLM(tokens: list, vocab):
    """
    :param tokens: 字符串列表
    :param vocab: 词表
    :return: 替换后的tokens, predict_positions_labels（预测位置和真实标签）
    """
    num_predictions = max(1, round(0.15 * (len(tokens) - 3)))  # 15%的词被替换
    candidate_predict_position = list(range(0, len(tokens)))
    random.shuffle(candidate_predict_position)
    predict_positions, predict_labels = [], []
    for predict_position in candidate_predict_position:
        if tokens[predict_position] == '<cls>' or tokens[predict_position] == '<cls>':
            continue
        if len(predict_positions) == num_predictions:
            break
        if random.random() < 0.8:
            mask_token = '<mask>'
        else:
            if random.random() < 0.5:
                mask_token = random.choice(vocab.idx_to_token)
            else:
                mask_token = tokens[predict_position]
        predict_positions.append(predict_position)
        predict_labels.append(vocab[tokens[predict_position]])
        tokens[predict_position] = mask_token
    return tokens, predict_positions, predict_labels


def get_bert_input(vocab, all_sentcences, max_len):
    """
    把数据转换成Bert的输入形式
    :param vocab: 词表
    :param all_sentcences: 所有的句子列表，每个元素也是一个列表，其中每个元素是一个词
    :param max_len: 序列最大长度
    :return: bert输入的数据
    """
    all_tokens, all_segments, valid_lens, all_is_next, all_predict_positions, all_mlm_weights, all_predict_labels = \
        [], [], [], [], [], [], []
    max_num_predictions = round(0.15 * max_len)
    for i in range(len(all_sentcences) - 1):
        token_a, token_b, is_next = get_next_sentence(all_sentcences[i], all_sentcences[i + 1],
                                                      all_sentcences[:i + 1] + all_sentcences[i + 2:])
        if len(token_a) + len(token_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(token_a, token_b)
        tokens, predict_positions, predict_labels = get_maskLM(tokens, vocab)
        # 长度不够的做填充
        all_tokens.append(vocab[tokens] + (max_len - len(tokens)) * [vocab['<pad>']])
        all_segments.append(vocab[segments] + (max_len - len(segments)) * [vocab['<pad>']])
        valid_lens.append(len(tokens))
        all_is_next.append(is_next)
        all_predict_positions.append(predict_positions + (max_num_predictions - len(predict_positions)) * [0])
        # all_mlm_weights后续计算损失的时候要用
        all_mlm_weights.append([1] * len(predict_positions) + (max_num_predictions - len(predict_positions)) * [0])
        all_predict_labels.append(predict_labels + (max_num_predictions - len(predict_labels)) * [0])

    return torch.tensor(all_tokens), torch.tensor(all_segments), torch.tensor(valid_lens), \
           torch.tensor(all_is_next, dtype=torch.long), torch.tensor(all_predict_positions), \
           torch.tensor(all_mlm_weights), torch.tensor(all_predict_labels, dtype=torch.long)


class WikiTextDataset(Dataset):
    """加载数据集"""

    def __init__(self, vocab, all_sentcences, max_len):
        self.all_tokens, self.all_segments, self.valid_lens, self.nsp_labels, self.all_predict_positions, \
        self.all_mlm_weights, self.mlm_labels = get_bert_input(vocab, all_sentcences, max_len)

    def __getitem__(self, index) -> T_co:
        return self.all_tokens[index], self.all_segments[index], self.valid_lens[index], self.all_predict_positions[
            index], \
               self.all_mlm_weights[index], self.nsp_labels[index], self.mlm_labels[index]

    def __len__(self):
        return len(self.all_tokens)


class BERTLoss(nn.Module):
    """计算BERT的损失"""

    def __init__(self):
        super(BERTLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, nsp_Y_hat, nsp_Y, mlm_Y_hat, mlm_Y, mlm_weights):
        nsp_loss = self.loss(nsp_Y_hat, nsp_Y).mean()
        mlm_loss = (self.loss(mlm_Y_hat.reshape(-1, mlm_Y_hat.shape[-1]), mlm_Y.reshape(-1))
                    * mlm_weights.reshape(-1)).sum() / mlm_weights.sum()
        all_loss = nsp_loss + mlm_loss
        return all_loss, nsp_loss, mlm_loss


if __name__ == '__main__':
    pass


    def train(net, train_iter, lr, num_epochs, target_vocab, device):
        loss = MaskCrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_tokens = 0
            for batch in train_iter:
                opt.zero_grad()
                x, x_valid_len, y, y_valid_len = [b.to(device) for b in batch]
                bos = torch.tensor([target_vocab["<bos>"]] * y.shape[0], device=device).reshape((-1, 1))
                dec_x = torch.cat((bos, y[:, :-1]), dim=1)
                y_hat, _ = net(x, dec_x, x_valid_len)
                l = loss(y_hat, y, y_valid_len)
                l.sum().backward()
                d2l.grad_clipping(net, 1)
                opt.step()
                epoch_loss += l.sum().item()
                num_tokens += y_valid_len.sum().item()
            loss_list.append(epoch_loss / num_tokens)
            if (epoch + 1) % 10 == 0:
                print(f"epoch:{epoch + 1}, loss:{epoch_loss / num_tokens : .3f} / token")


    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])


    import d2l.torch as d2l

    num_hiddens, num_lays, dropout, batch_size, num_steps = 32, 2, 0.1, 256, 10
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_hiddens, num_heads = 64, 4
    norm_shape = [32]
    loss_list = []
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = TransformerEncoder(len(src_vocab), num_hiddens, num_lays, num_heads, norm_shape, ffn_num_hiddens, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), num_hiddens, num_lays, num_heads, norm_shape, ffn_num_hiddens, dropout)
    net = EncoderDecoder(encoder, decoder)
    net.apply(xavier_init_weights)
    net.to(device)
    train(net, train_iter, lr, num_epochs, tgt_vocab, device)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 3))
    plt.plot(range(num_epochs), loss_list)
    plt.grid(True)
    # plt.xlim((0, 500))
    plt.show()
