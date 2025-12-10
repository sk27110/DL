import torch
import torch.nn as nn
import torchvision.models as models
import math


class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.resnet50(weights="DEFAULT", progress=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
                
        return self.dropout(self.relu(features))
    


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

    def forward(self, features, captions, tgt_mask, tgt_key_padding_mask):


        captions = captions.transpose(0, 1)
        embeddings = self.dropout(self.embed(captions) * math.sqrt(self.embed_size))
        embeddings = self.pos_encoder(embeddings)

     
        memory = features.unsqueeze(0) 

        output = self.transformer_decoder(
            tgt=embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None 
        ) 

        output = self.linear(output)
        output = output.transpose(0, 1)
        return output
    

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, vocab_size, num_layers, dropout=0.4):
        super().__init__()
        self.encoder = Encoder(embed_size) 
        self.decoder = Decoder(embed_size, num_heads, num_layers, vocab_size, dropout)
    
    def forward(self, images, captions, tgt_mask, tgt_key_padding_mask):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, tgt_mask, tgt_key_padding_mask)
        return outputs