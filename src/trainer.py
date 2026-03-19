import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

trainer_print_prefix = "trainer --> "


class MemorizationTrainer:
    def __init__(self, model, prefix_len, learning_rate=1e-3, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.prefix_len = prefix_len

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.0)

    def train_to_memorize(self, dataloader, max_epochs=500):
        self.model.train()
        max_accuracy = 0.0
        best_loss = float('inf')
        patience_counter = 0
        patient_threshold = 100

        epoch_iterator = tqdm(itertools.count(), desc=trainer_print_prefix, leave=False, unit="epoch")
        for epoch in epoch_iterator:
            # total_loss = 0.0
            correct_sequences = 0
            total_sequences = 0

            for batch in dataloader:
                seqs = batch[0].to(self.device)

                inputs = seqs[:, :-1]
                targets = seqs[:, 1:]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                raw_loss = self.criterion(outputs_flat, targets_flat)
                raw_loss = raw_loss.view(targets.shape)

                mask = torch.zeros_like(targets, dtype=torch.bool)
                mask[:, (self.prefix_len - 1):] = True

                masked_loss = raw_loss[mask].mean()
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                # total_loss += masked_loss.item()

                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=-1)
                    payload_preds = predictions[:, self.prefix_len - 1:]
                    payload_targets = targets[:, self.prefix_len - 1:]
                    sequence_matches = (payload_preds == payload_targets).all(dim=1)
                    correct_sequences += sequence_matches.sum().item()
                    total_sequences += seqs.size(0)

            accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
            max_accuracy = max(accuracy, max_accuracy)
            current_loss = masked_loss.item()
            
            epoch_iterator.set_postfix(accuracy=f"{max_accuracy*100:.2f}%", loss=f"{current_loss:.4f}", patience=patience_counter)

            if max_accuracy >= 1.0:
                break

            if current_loss < (best_loss - 1e-4):
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patient_threshold:
                # print(f"Early stopping at epoch {epoch} with accuracy {max_accuracy:.2f}")
                break

        # del outputs, outputs_flat, targets_flat, raw_loss, masked_loss
        epoch_iterator.close()

        return max_accuracy
