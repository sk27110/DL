import torch
import torch.nn as nn
import torchvision.models as models
import math
import torch.nn.functional as F


def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    b, l, d = u.shape
    n = A.shape[1]
    Delta = delta
    if delta_bias is not None:
        Delta = Delta + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        Delta = F.softplus(Delta)
    A_bar = torch.exp(Delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (b, l, d, n)
    B_bar = Delta.unsqueeze(-1) * B.unsqueeze(2)  # (b, l, d, n)
    h = torch.zeros(b, d, n, device=u.device)
    ys = torch.zeros(b, l, d, device=u.device)
    for i in range(l):
        h = A_bar[:, i] * h + B_bar[:, i] * u[:, i].unsqueeze(-1)
        y = (h * C[:, i].unsqueeze(1)).sum(-1)
        if D is not None:
            y += D * u[:, i]
    ys[:, i] = y
    return ys


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv, bias=True, groups=self.d_inner, padding=self.d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # Initialize dt_proj bias
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_proj.bias.data = torch.log(dt)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)  # (b, l, 2 * d_inner)
        x, res = x_and_res.chunk(2, dim=-1)
        x = x.permute(0, 2, 1)  # (b, d_inner, l)
        x = self.conv1d(x)[:, :, :l]
        x = x.permute(0, 2, 1)  # (b, l, d_inner)
        x = self.act(x)
        x_db = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (b, l, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        y = selective_scan(x, dt, A, B, C, self.D, delta_softplus=True)
        y = y * self.act(res)
        y = self.out_proj(y)
        return y


class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False, num_encoder_layers=3, num_heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(weights="DEFAULT")

        for param in resnet.parameters():
            param.requires_grad = False

        if train_CNN:
            for param in resnet.layer3.parameters():
                param.requires_grad = True
            for param in resnet.layer4.parameters():
                param.requires_grad = True


        modules = list(resnet.children())[:-2]
        
        self.cnn = nn.Sequential(*modules)
        self.conv_proj = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pos_encoder = PositionalEncoding(embed_size, max_len=49)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True  
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        nn.init.xavier_uniform_(self.conv_proj.weight)
        nn.init.zeros_(self.conv_proj.bias)

    def forward(self, images):

        features = self.cnn(images)
        features = self.adaptive_pool(features) 
        features = self.conv_proj(features) 
        features = self.relu(features)
        features = self.dropout(features)

        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.pos_encoder(features.transpose(0, 1)).transpose(0, 1)
        features = self.transformer_encoder(features) 

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


class MambaDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.mamba = Mamba(embed_size)
        self.cross_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        x = tgt
        # Mamba as self "attention"
        mamba_input = x.permute(1, 0, 2)  # (seq, batch, d) -> (batch, seq, d)
        x2 = self.mamba(mamba_input)
        x2 = x2.permute(1, 0, 2)  # back to (seq, batch, d)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        # Cross attention
        x2 = self.cross_attn(x, memory, memory, attn_mask=None, key_padding_mask=None)[0]
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        # FF
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)

        self.layers = nn.ModuleList([MambaDecoderLayer(embed_size, num_heads, embed_size * 4, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.embed.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features, captions, tgt_mask, tgt_key_padding_mask):
        # features: (batch, mem_len, embed_size) but need to permute to (mem_len, batch, embed_size)
        memory = features.permute(1, 0, 2)
        captions = captions.transpose(0, 1)  # (batch, seq) -> (seq, batch)

        embeddings = self.embed(captions) * math.sqrt(self.embed_size)
        embeddings = self.pos_encoder(embeddings)
        output = self.dropout(embeddings)

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.linear(output)
        output = output.transpose(0, 1)  # (seq, batch, vocab) -> (batch, seq, vocab)

        return output


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, vocab_size, num_layers, dropout=0.4, num_encoder_layers=3, train_CNN = False):
        super().__init__()
        self.encoder = Encoder(embed_size, train_CNN=train_CNN, num_encoder_layers=num_encoder_layers, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(embed_size, num_heads, num_layers, vocab_size, dropout)

        self.embed_size = embed_size
        self.vocab_size = vocab_size

    def forward(self, images, captions, tgt_mask, tgt_key_padding_mask):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, tgt_mask, tgt_key_padding_mask)
        return outputs

    def generate(self, images, max_len=50, start_token=1, end_token=2):
        with torch.no_grad():
            batch_size = images.size(0)
            features = self.encoder(images)
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=images.device)

            for _ in range(max_len):
                seq_len = generated.size(1)
                tgt_mask = self._generate_square_subsequent_mask(seq_len).to(images.device)
                tgt_key_padding_mask = None

                logits = self.decoder(features, generated, tgt_mask, tgt_key_padding_mask)
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)

                if torch.all(next_token == end_token):
                    break

            return [seq.tolist() for seq in generated]

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask