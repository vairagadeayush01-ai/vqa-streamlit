import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = resnet50(pretrained=True)
        
        # Remove avgpool and fc
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        feat = self.backbone(x)
        # feat shape: (batch_size, 2048, 7, 7)
        return feat

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        # x shape: (batch_size, 2048, 7, 7)
        
        attn_map = self.conv(x)
        # attn_map shape: (batch_size, 1, 7, 7)
        
        attn_map = attn_map.view(attn_map.size(0), -1)
        # shape: (batch_size, 49)
        
        attn_weights = torch.softmax(attn_map, dim=1)
        # shape: (batch_size, 49)
        
        attn_weights = attn_weights.view(x.size(0), 1, 7, 7)
        # shape: (batch_size, 1, 7, 7)
        
        weighted_feat = x * attn_weights
        # shape: (batch_size, 2048, 7, 7)
        
        img_embed = weighted_feat.sum(dim=[2, 3])
        # img_embed shape: (batch_size, 2048)
        
        return img_embed

from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        # input_ids shape: (batch_size, max_len)
        # attention_mask shape: (batch_size, max_len)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_embed = outputs.last_hidden_state[:, 0, :]
        # cls_embed shape: (batch_size, 768)
        
        return cls_embed

class GatedFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim):
        super().__init__()
        
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, img_feat, txt_feat):
        # img_feat shape: (batch_size, 2048)
        # txt_feat shape: (batch_size, 768)
        
        img_h = self.img_proj(img_feat)
        # shape: (batch_size, hidden_dim)
        
        txt_h = self.txt_proj(txt_feat)
        # shape: (batch_size, hidden_dim)
        
        concat = torch.cat([img_h, txt_h], dim=1)
        # shape: (batch_size, hidden_dim * 2)
        
        gate = torch.sigmoid(self.gate(concat))
        # shape: (batch_size, hidden_dim)
        
        fused = gate * img_h + (1 - gate) * txt_h
        # shape: (batch_size, hidden_dim)

        return fused

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        self.image_encoder = ResNetBackbone()
        self.spatial_attn = SpatialAttention(in_channels=2048)
        self.text_encoder = TextEncoder()
        
        self.fusion = GatedFusion(
            img_dim=2048,
            txt_dim=768,
            hidden_dim=1024
        )
        
        self.classifier = nn.Linear(1024, num_answers)

    def forward(self, images, input_ids, attention_mask):
        # images shape: (batch_size, 3, 224, 224)
        # input_ids shape: (batch_size, max_len)
        # attention_mask shape: (batch_size, max_len)
        
        img_feat_map = self.image_encoder(images)
        # shape: (batch_size, 2048, 7, 7)
        
        img_feat = self.spatial_attn(img_feat_map)
        # shape: (batch_size, 2048)
        
        txt_feat = self.text_encoder(input_ids, attention_mask)
        # shape: (batch_size, 768)
        
        fused = self.fusion(img_feat, txt_feat)
        # shape: (batch_size, 1024)
        
        logits = self.classifier(fused)
        # logits shape: (batch_size, 3000)
        
        return logits
