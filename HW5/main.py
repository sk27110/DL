from src.model import EncoderDecoder
from src.dataset import get_datasets
from src.trainer import Trainer
from src.utils import CollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def main():
    train, val, test = get_datasets()
    pad_idx = train.vocab.stoi["<PAD>"]
    batch_size = 64
    train_loader = DataLoader(
        train, 
        batch_size=batch_size, 
        num_workers=2, 
        shuffle=True, 
        pin_memory=True,
        collate_fn=CollateFn(pad_idx)
    )

    val_loader = DataLoader(
        val, 
        batch_size=batch_size, 
        num_workers=2, 
        shuffle=False,
        pin_memory=True, 
        collate_fn=CollateFn(pad_idx)
    )

    embed_size = 512
    hidden_size = 512
    num_layers = 3
    vocab_size = len(train.vocab)
    num_epochs = 100
    num_heads = 8
    learning_rate = 1e-4

    model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers)

    criterion = nn.CrossEntropyLoss(ignore_index=train.vocab.stoi["<PAD>"], label_smoothing=0.03)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)

    train_dataset_len = len(train_loader.dataset) 

    total_steps = (train_dataset_len // batch_size) * num_epochs
    total_steps = len(train_loader) * num_epochs

    warmup_steps = 500         


    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,         
        num_training_steps=total_steps,          
        num_cycles=0.5,                         
        last_epoch=-1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path="./best_model.pth",
        pad_idx=pad_idx,
        patience=5,
        max_gen_len=20,
        tokenizer=train.vocab
            )
    
    trainer.train()


if __name__ == "__main__":
    main()