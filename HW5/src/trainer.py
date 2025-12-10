import wandb
import torch
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, save_path,
                 patience=5, pad_idx=0, end_idx = 2, max_gen_len=20, tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_path = save_path
        self.patience = patience
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        self.max_gen_len = max_gen_len
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    def _get_tgt_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)
        return mask

    def _get_padding_mask(self, captions):
        return (captions == self.pad_idx).to(self.device)

    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            features, captions = batch
            features, captions = features.to(self.device), captions.to(self.device)

            tgt_input = captions[:, :-1]
            targets = captions[:, 1:]

            tgt_mask = self._get_tgt_mask(tgt_input.shape[1])
            tgt_key_padding_mask = self._get_padding_mask(tgt_input)

            outputs = self.model(features, tgt_input, tgt_mask, tgt_key_padding_mask)

            loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_loader)
        return avg_train_loss

    def _validate(self):
        self.model.eval()
        val_loss = 0
        examples = []

        with torch.no_grad():
            for batch_idx, (features, captions) in enumerate(tqdm(self.val_loader, desc="Validation")):
                features, captions = features.to(self.device), captions.to(self.device)

                tgt_input = captions[:, :-1]
                targets = captions[:, 1:]

                tgt_mask = self._get_tgt_mask(tgt_input.shape[1])
                tgt_key_padding_mask = self._get_padding_mask(tgt_input)

                outputs = self.model(features, tgt_input, tgt_mask, tgt_key_padding_mask)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
                val_loss += loss.item()


                if batch_idx < 3:
                    for i in range(min(2, features.size(0))):
                        gen_ids = self.generate(features[i].unsqueeze(0), sos_idx=self.tokenizer.stoi["<START>"])
                        gen_tokens = [self.tokenizer.itos[idx] for idx in gen_ids if idx != self.tokenizer.stoi["<PAD>"]]
                        true_tokens = [self.tokenizer.itos[idx] for idx in captions[i].cpu().numpy() if idx != self.tokenizer.stoi["<PAD>"]]

                        examples.append({
                            "prediction": " ".join(gen_tokens),
                            "ground_truth": " ".join(true_tokens)
                        })

            log_dict = {}
            for idx, ex in enumerate(examples):
                log_dict[f"example_{idx}_prediction"] = ex["prediction"]
                log_dict[f"example_{idx}_ground_truth"] = ex["ground_truth"]

            wandb.log(log_dict)

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss, examples

    @torch.no_grad()
    def generate(self, features, sos_idx):
        self.model.eval()
        features = features.to(self.device)
        generated = [sos_idx]
        for _ in range(self.max_gen_len):
            tgt_input = torch.tensor(generated, device=self.device).unsqueeze(0) 
            tgt_mask = self._get_tgt_mask(tgt_input.shape[1])
            output = self.model(features, tgt_input, tgt_mask, None)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == self.end_idx:
                break
        return generated

    def train(self, num_epochs=10):
        best_val = np.inf
        wait = 0

        wandb.init(project="transformer_image_captioning")

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, examples = self._validate()

            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), self.save_path)
                artifact = wandb.Artifact('my_model', type='model')  # Имя артефакта и тип
                artifact.add_file(self.save_path)  # Добавь локальный файл модели (например, 'best_model.pth')
                wandb.log_artifact(artifact)
                best_checkpoint = self.save_path
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            for idx, ex in enumerate(examples):
                log_dict[f"example_{idx}_prediction"] = ex["prediction"]
                log_dict[f"example_{idx}_ground_truth"] = ex["ground_truth"]

            wandb.log(log_dict, step=epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            for ex in examples:
                print(f"Pred: {ex['prediction']}")
                print(f"GT  : {ex['ground_truth']}")
                print("-"*40)

        print(f"Training finished. Best checkpoint saved at: {best_checkpoint}")
        wandb.finish()
