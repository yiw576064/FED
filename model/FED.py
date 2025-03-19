import torch
import timm
from transformers import RobertaModel
import torch.nn.functional as F
import copy
import model

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions

class FED(torch.nn.Module):
    # define model elements
    def __init__(self, bertl_text, vit, opt):
        super(FED, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit = vit
        if not self.opt["finetune"]:
            freeze_layers(self.bertl_text)
            freeze_layers(self.vit)
        assert ("input1" in opt)
        assert ("input2" in opt)
        assert ("input3" in opt)
        self.input1 = opt["input1"]
        self.input2 = opt["input2"]
        self.input3 = opt["input3"]

        self.trar = model.TRAR.FED(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"], 2)
        )

        self.classifier_fuse = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768, 2)
        ) 
        self.classifier_text = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768, 2)
        ) 
        self.classifier_image = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768, 2)
        )  

        self.loss_fct = nn.CrossEntropyLoss()  

        self.guide_attention_layer = GuideAttentionLayer(batch_size=32, text_seq_len=opt['len'],
                                                         text_hidden_dim=opt['mlp_size'],
                                                         image_block_num=opt['IMG_SCALE'] * opt['IMG_SCALE'],
                                                         image_hidden_dim=opt['mlp_size'], use_type=3, use_source=1)
        self.guide_attention = GuideAttention(batch_size=32, text_seq_len=opt['len'],
                                                         text_hidden_dim=opt['mlp_size'],
                                                         image_block_num=opt['IMG_SCALE'] * opt['IMG_SCALE'],
                                                         image_hidden_dim=opt['mlp_size'], use_source=1)
        self.tradition_attention_layer = TraditionalAttentionLayer(text_seq_len=opt['len'],
                                                                   text_hidden_dim=opt['mlp_size'],
                                                                   image_block_num=opt['IMG_SCALE'] * opt['IMG_SCALE'],
                                                                   image_hidden_dim=opt['mlp_size'], use_type=3)

    def vit_forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:, 1:]

    # forward propagate input
    def forward(self, input):
        bert_embed_text = self.bertl_text.embeddings(input_ids=input[self.input1])
        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text

        img_feat = self.vit_forward(input[self.input2])
        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text, input[self.input3].unsqueeze(1).unsqueeze(2))

        # Feature fusion
        attention_mask = torch.cat((torch.ones(bert_text.shape[0], bert_text.shape[1]).to(bert_text.device),
                                    torch.ones(img_feat.shape[0], img_feat.shape[1]).to(img_feat.device)), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        fuse_hiddens, _ = self.trans(torch.cat((img_feat, bert_text), dim=1), extended_attention_mask,
                                     output_all_encoded_layers=False)
        fuse_hiddens = fuse_hiddens[-1]
        new_text_features = fuse_hiddens[:, bert_text.shape[1]:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=input[self.input1].device), input[self.input1].to(
                torch.int).argmax(dim=-1)
        ]

        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)
        att = F.softmax(torch.stack((text_weight, image_weight), dim=-1), dim=-1)
        tw, iw = att.split([1, 1], dim=-1)
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature

        logits_fuse = self.classifier_fuse(fuse_feature)
        logits_text = self.classifier_text(new_text_feature)
        logits_image = self.classifier_image(new_image_feature)

        fuse_score = F.softmax(logits_fuse, dim=-1)
        text_score = F.softmax(logits_text, dim=-1)
        image_score = F.softmax(logits_image, dim=-1)

        score = fuse_score + text_score + image_score

        result = self.sigm(score)

        return result, lang_emb, img_emb


def build_FED(opt, requirements):
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"] = "vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return FED(bertl_text, vit, opt)