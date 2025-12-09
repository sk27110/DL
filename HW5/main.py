from src.model import EncoderDecoder
from src.dataset import get_datasets
from src.trainer import Trainer
from src.utils import CollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def main():
    train, val, test = get_datasets()
    pad_idx = train.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        train, 
        batch_size=16, 
        num_workers=2, 
        shuffle=True, 
        pin_memory=True,
        collate_fn=CollateFn(pad_idx)
    )

    val_loader = DataLoader(
        val, 
        batch_size=64, 
        num_workers=2, 
        shuffle=False,
        pin_memory=True, 
        collate_fn=CollateFn(pad_idx)
    )

    embed_size = 512
    hidden_size = 512
    num_layers = 3
    vocab_size = len(train.vocab)
    num_epochs = 10
    num_heads = 8
    learning_rate = 0.0005

    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers)

    criterion = nn.CrossEntropyLoss(ignore_index=train.vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         
        factor=0.5,         
        patience=3,         
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path="./",
        pad_idx=pad_idx,
        patience=5,
        max_gen_len=20,
        tokenizer=train.vocab
            )
    
    trainer.train()

