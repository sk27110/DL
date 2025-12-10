import torch
import torch.nn as nn
import torchvision.models as models
import math


class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        
        # Используем ResNet50
        resnet = models.resnet50(weights="DEFAULT")
        
        # Замораживаем все слои кроме последних блоков
        if not train_CNN:
            for param in resnet.parameters():
                param.requires_grad = False
        
        # Размораживаем последние блоки если нужно
        if train_CNN:
            # Размораживаем layer4 и layer3
            for param in resnet.layer3.parameters():
                param.requires_grad = True
            for param in resnet.layer4.parameters():
                param.requires_grad = True
        
        # Берем до последнего conv слоя (до avgpool)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        
        # Проекция feature map в embed_size
        self.conv_proj = nn.Conv2d(2048, embed_size, kernel_size=1)
        
        # Adaptive pooling для фиксированного размера
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Инициализация проекции
        nn.init.xavier_uniform_(self.conv_proj.weight)
        nn.init.zeros_(self.conv_proj.bias)
    
    def forward(self, images):
        # Получаем feature map: (B, 2048, H/32, W/32)
        features = self.cnn(images)
        
        # Приводим к фиксированному размеру и проецируем
        features = self.adaptive_pool(features)  # (B, 2048, 7, 7)
        features = self.conv_proj(features)     # (B, embed_size, 7, 7)
        features = self.relu(features)
        features = self.dropout(features)
        
        # Преобразуем в (B, embed_size, 49) для трансформера
        batch_size, embed_size = features.size(0), features.size(1)
        features = features.view(batch_size, embed_size, -1)  # (B, embed_size, 49)
        features = features.permute(0, 2, 1)  # (B, 49, embed_size)
        
        return features
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, dropout):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        
        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=False
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        
        # Инициализация
        self._init_weights()
    
    def _init_weights(self):
        # Xavier инициализация для лучшей сходимости
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Особенная инициализация для embedding и linear
        nn.init.normal_(self.embed.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, features, captions, tgt_mask, tgt_key_padding_mask):
        # features: (B, 49, embed_size) -> (49, B, embed_size)
        features = features.permute(1, 0, 2)
        
        # captions: (B, seq_len) -> (seq_len, B)
        captions = captions.transpose(0, 1)
        
        # Embedding с масштабированием
        embeddings = self.embed(captions) * math.sqrt(self.embed_size)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transformer Decoder
        output = self.transformer_decoder(
            tgt=embeddings,
            memory=features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Линейный слой
        output = self.linear(output)
        output = output.transpose(0, 1)  # (B, seq_len, vocab_size)
        
        return output
    

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, vocab_size, num_layers, dropout=0.4):
        super().__init__()
        self.encoder = Encoder(embed_size, train_CNN=False)
        self.decoder = Decoder(embed_size, num_heads, num_layers, vocab_size, dropout)
    
    def forward(self, images, captions, tgt_mask, tgt_key_padding_mask):
        features = self.encoder(images)  # (B, 49, embed_size)
        outputs = self.decoder(features, captions, tgt_mask, tgt_key_padding_mask)
        return outputs