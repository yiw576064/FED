import torch
from torch import nn

import math

from model.attention.GuideAttention import GuideAttention

'''
分别自注意力
'''


class GuideAttentionLayer:
    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim, use_type,
                 use_source):
        self.batch_size = batch_size
        self.text_seq_len = text_seq_len
        self.text_hidden_dim = text_hidden_dim
        self.image_block_num = image_block_num
        self.image_hidden_dim = image_hidden_dim
        self.use_type = use_type
        self.strategies = {
            0: TextGuideAttention(batch_size, text_seq_len, text_hidden_dim),
            1: ImageGuideAttention(batch_size, image_block_num, image_hidden_dim),
            2: BothGuideAttention(batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim),
            3: GuideAttention(batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim, use_source)
        }

    def process(self, text_feature=None, image_feature=None):
        '''
        :param use_type: 标注使用的方法类型
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        '''

        return self.strategies.get(self.use_type)(text_feature, image_feature)


'''
分别自注意力
'''


class BothGuideAttention(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(BothGuideAttention, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.text_sparse_attention = LocalAttention(text_seq_len, text_hidden_dim)
        self.image_sparse_attention = LocalAttention(image_block_num, image_hidden_dim)

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim)).to(self.device)
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim)).to(self.device)

        # norm
        self.text_norm = nn.LayerNorm(
            normalized_shape=[batch_size, text_seq_len, text_hidden_dim],
            eps=1e-6,
            device=self.device)
        self.image_norm = nn.LayerNorm(
            normalized_shape=[batch_size, image_block_num, image_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 先对图片进行处理
        # 1. sparse attention
        image_out = self.image_sparse_attention(image_feature)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)
        # 2. MLP
        image_out = self.image_out(image_out)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)

        # 再对文本处理
        # 1. sparse attention
        text_out = self.text_sparse_attention(text_feature)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)
        # 2. MLP out
        text_out = self.text_out(text_out)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)

        return text_out, image_out

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)


'''
文本自注意力
'''


class TextGuideAttention(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim):
        super(TextGuideAttention, self).__init__()
        self.batch_size = batch_size

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # sparse attention
        self.text_sparse_attention = LocalAttention(text_seq_len, text_hidden_dim)

        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim)).to(self.device)

        # norm
        self.text_norm = nn.LayerNorm(
            normalized_shape=[batch_size, text_seq_len, text_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 1. sparse attention
        text_out = self.text_sparse_attention(text_feature)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)
        # 2. MLP out
        text_out = self.text_out(text_out)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)

        return text_out, image_feature

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)


'''
图片自注意力
'''


class ImageGuideAttention(nn.Module):

    def __init__(self, batch_size, image_block_num, image_hidden_dim):
        super(ImageGuideAttention, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.image_sparse_attention = LocalAttention(image_block_num, image_hidden_dim)

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # MLP
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim)).to(self.device)

        # norm
        self.image_norm = nn.LayerNorm(
            normalized_shape=[batch_size, image_block_num, image_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 1. sparse attention
        image_out = self.image_sparse_attention(image_feature)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)
        # 2. MLP
        image_out = self.image_out(image_out)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)

        return text_feature, image_out

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)



class LocalAttention(nn.Module):
    def __init__(self, two_pos, hidden_dim, window_size=2, sparsity=2):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.sparsity = sparsity
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # 图片作为Q
        # 定义 Q, K, V 的线性映射
        self.q_linear = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim).to(self.device)

        # 定义局部注意力分布
        attention_mask = torch.zeros(two_pos, two_pos)
        attention_mask = attention_mask.to(self.device)
        for i in range(two_pos):
            start = max(0, i - self.window_size)
            end = min(two_pos, i + self.window_size + 1)
            attention_mask[i, start:end] = 1

        # 计算注意力权重
        self.attention_weights = torch.softmax(attention_mask, dim=-1)

    def forward(self, feature):
        d_k = feature.size(-1)
        # 线性映射得到 Q, K, V
        q = self.q_linear(feature)
        k = self.k_linear(feature)
        v = self.v_linear(feature)

        # 这里单独写出来batch size是因为存在数据集最后一部分没有够一个batch size
        batch_size = feature.size(0)
        attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        # 构造一个全0矩阵，然后取attention_weights中取tok（2 * window_size  + 1 + k）个，
        num_nonzero = int(feature.size(1) // self.sparsity) + 2 * self.window_size
        topk_scores, topk_indices = torch.topk(attention_weights, num_nonzero, dim=-1)

        # 构建稀疏注意力矩阵
        sparse_attention = torch.zeros_like(attention_weights).to(self.device)
        sparse_attention.scatter_(-1, topk_indices, topk_scores)
        output = torch.bmm(sparse_attention, k)  # (batch_size, seq_len, hidden_dim)
        output = torch.bmm(q, output.transpose(2, 1)) / math.sqrt(d_k)  # (batch_size, 1, seq_len)
        output = torch.bmm(torch.softmax(output, dim=-1), v)  # (batch_size, 1, text_hidden_dim)

        return output


if __name__ == '__main__':
    text_input = torch.randn(8, 128, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    image_input = torch.randn(8, 197, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    attr = GuideAttentionLayer(8, 128, 768, 197, 768, 3, 1)
    res1, res2 = attr.process(text_input, image_input)
    print(res1.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
    print(res2.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
