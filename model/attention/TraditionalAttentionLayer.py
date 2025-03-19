import math

import torch
import torch.nn as nn


class TraditionalAttentionLayer:
    def __init__(self, text_seq_len=None, text_hidden_dim=None, image_block_num=None, image_hidden_dim=None,
                 use_type=None):
        super(TraditionalAttentionLayer, self).__init__()
        self.text_seq_len = text_seq_len
        self.text_hidden_dim = text_hidden_dim
        self.image_block_num = image_block_num
        self.image_hidden_dim = image_hidden_dim
        self.use_type = use_type
        self.strategies = {
            0: TextTraditionalSelfAttention(text_hidden_dim),
            1: ImageTraditionalSelfAttention(image_hidden_dim),
            2: BothTraditionalSelfAttention(text_hidden_dim, image_hidden_dim),
            3: TraditionalCrossAttention(text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim)
        }

    def process(self, text_feature=None, image_feature=None):
        '''
        :param use_type: 标注使用的方法类型
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        '''

        return self.strategies.get(self.use_type)(text_feature, image_feature)


'''
传统交叉注意力计算
'''


class TraditionalCrossAttention(nn.Module):
    def __init__(self, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(TraditionalCrossAttention, self).__init__()
        self.text_seq_len = text_seq_len
        self.image_block_num = image_block_num
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.text_attention = nn.MultiheadAttention(embed_dim=text_hidden_dim, num_heads=1).to(self.device)
        self.image_attention = nn.MultiheadAttention(embed_dim=image_hidden_dim, num_heads=1).to(self.device)


        # 定义文本长度和图片块数互换
        self.seq_len_to_block = nn.Linear(text_seq_len, image_block_num).to(self.device)
        self.seq_len_from_block = nn.Linear(image_block_num, text_seq_len).to(self.device)
        # 定义图片块数和文本长度互换
        self.block_to_seq_len = nn.Linear(image_block_num, text_seq_len).to(self.device)
        self.block_from_seq_len = nn.Linear(text_seq_len, image_block_num).to(self.device)

    def forward(self, text_feature, image_feature):
        '''
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        '''
        q_text = self.seq_len_to_block(text_feature.transpose(2, 1)).transpose(2, 1)
        q_image = self.block_to_seq_len(image_feature.transpose(2, 1)).transpose(2, 1)

        output_text, _ = self.text_attention(q_text, image_feature, image_feature)
        output_image, _ = self.image_attention(q_image, text_feature, text_feature)

        output_text = self.seq_len_from_block(output_text.transpose(2, 1)).transpose(2, 1)
        output_image = self.block_from_seq_len(output_image.transpose(2, 1)).transpose(2, 1)

        return output_text, output_image


'''
对文本和图片各自做传统注意力计算
'''


class BothTraditionalSelfAttention(nn.Module):
    def __init__(self, text_hidden_dim, image_hidden_dim):
        super(BothTraditionalSelfAttention, self).__init__()
        self.text_attn = TraditionalSelfAttention(text_hidden_dim)
        self.image_attn = TraditionalSelfAttention(image_hidden_dim)

    def forward(self, text_feature, image_feature):
        return self.text_attn(text_feature), self.image_attn(image_feature)


'''
文本传统自注意力计算
'''


class TextTraditionalSelfAttention(nn.Module):
    def __init__(self, text_hidden_dim):
        super(TextTraditionalSelfAttention, self).__init__()
        self.text_attn = TraditionalSelfAttention(text_hidden_dim)

    def forward(self, text_feature, image_feature):
        return self.text_attn(text_feature), image_feature


'''
图片传统自注意力计算
'''


class ImageTraditionalSelfAttention(nn.Module):
    def __init__(self, image_hidden_dim):
        super(ImageTraditionalSelfAttention, self).__init__()
        self.image_attn = TraditionalSelfAttention(image_hidden_dim)

    def forward(self, text_feature, image_feature):
        return text_feature, self.image_attn(image_feature)


'''
做传统自注意力计算
'''


class TraditionalSelfAttention(nn.Module):
    def __init__(self, text_hidden_dim):
        super(TraditionalSelfAttention, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # 定义 Q, K, V 的线性映射
        self.q_linear = nn.Linear(text_hidden_dim, text_hidden_dim).to(self.device)
        self.k_linear = nn.Linear(text_hidden_dim, text_hidden_dim).to(self.device)
        self.v_linear = nn.Linear(text_hidden_dim, text_hidden_dim).to(self.device)

    def forward(self, feature):
        query, key, value = self.q_linear(feature), self.k_linear(feature), self.v_linear(feature)
        d_k = feature.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        return torch.bmm(torch.softmax(scores, dim=-1), value)


if __name__ == '__main__':
    text_input = torch.randn(8, 128, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    image_input = torch.randn(8, 197, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    attr = TraditionalAttentionLayer(128, 768, 197, 768, 3)
    res1, res2 = attr.process(text_input, image_input)
    print(res1.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
    print(res2.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
