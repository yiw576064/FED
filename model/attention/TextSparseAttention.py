import math

import torch
import torch.nn as nn

'''
稀疏注意力计算
当前需要注意window_size的设置是否合理
'''


class TextSparseAttention(nn.Module):
    def __init__(self, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim,
                 window_size=1, sparsity=2):
        super(TextSparseAttention, self).__init__()
        self.window_size = window_size
        self.text_seq_len = text_seq_len
        self.image_block_num = image_block_num
        self.sparsity = sparsity
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # 文本作为Q
        # 定义 Q, K, V 的线性映射
        self.q_linear = nn.Linear(text_hidden_dim, text_hidden_dim).to(self.device)
        self.k_linear = nn.Linear(image_hidden_dim, image_hidden_dim).to(self.device)
        self.v_linear = nn.Linear(image_hidden_dim, image_hidden_dim).to(self.device)

        # 定义文本长度到图片块数
        self.seq_len_to_block = nn.Linear(text_seq_len, image_block_num).to(self.device)

        # 定义局部注意力分布
        attention_mask = torch.zeros(self.text_seq_len, self.text_seq_len)

        attention_mask = attention_mask.to(self.device)
        for i in range(self.text_seq_len):
            start = max(0, i - self.window_size)
            end = min(self.text_seq_len, i + self.window_size + 1)
            attention_mask[i, start:end] = 1

        # 计算注意力权重
        self.attention_weights = torch.softmax(attention_mask, dim=-1)

    def forward(self, text_feature, image_feature):
        '''
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        '''

        d_k = text_feature.size(-1)
        # 线性映射得到 Q, K, V
        q = self.q_linear(text_feature)
        k = self.k_linear(image_feature)
        v = self.v_linear(image_feature)

        # 这里单独写出来batch size是因为存在数据集最后一部分没有够一个batch size
        batch_size = text_feature.size(0)
        attention_weights = self.attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        attention_weights = self.seq_len_to_block(attention_weights)
        # 构造一个全0矩阵，然后取attention_weights中取tok（2 * window_size  + 1 + k）个，
        num_nonzero = int(self.image_block_num // self.sparsity) + 2 * self.window_size
        topk_scores, topk_indices = torch.topk(attention_weights, num_nonzero, dim=-1)

        # 构建稀疏注意力矩阵
        sparse_attention = torch.zeros_like(attention_weights).to(text_feature.device)
        sparse_attention.scatter_(-1, topk_indices, topk_scores)
        output = torch.bmm(sparse_attention, k)  # (batch_size, seq_len, hidden_dim)
        output = torch.bmm(q, output.transpose(2, 1)) / math.sqrt(d_k)  # (batch_size, seq_len, seq_len)
        output = self.seq_len_to_block(output)
        output = torch.bmm(torch.softmax(output, dim=-1), v)  # (batch_size, seq_len, hidden_dim)

        return output

    def attention_weight(self, text_feature, image_feature):
        '''
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        '''

        # 线性映射得到 Q, K, V
        q = self.q_linear(text_feature)
        k = self.k_linear(image_feature)
        v = self.v_linear(image_feature)

        # 定义局部注意力分布
        attention_mask = torch.zeros(self.text_seq_len, self.text_seq_len)
        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        attention_mask = attention_mask.to(device)
        for i in range(self.text_seq_len):
            start = max(0, i - self.window_size)
            end = min(self.text_seq_len, i + self.window_size + 1)
            attention_mask[i, start:end] = 1

        # 计算注意力权重
        attention_weights = torch.softmax(attention_mask, dim=-1)

        # 加权求和
        # 这里单独写出来batch size是因为存在数据集最后一部分没有够一个batch size
        batch_size = text_feature.size(0)
        attention_weights = attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        attention_weights = self.seq_len_to_block(attention_weights)
        output = torch.bmm(attention_weights, k)  # (batch_size, seq_len, hidden_dim)
        output = torch.bmm(q, output.transpose(2, 1))  # (batch_size, seq_len, seq_len)
        return self.seq_len_to_block(output)
