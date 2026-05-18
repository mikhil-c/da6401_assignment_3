"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            pad_positions = (target == self.pad_idx)
            smooth_dist[pad_positions] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad = (~pad_positions).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} {'Train' if is_train else 'Val'}", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad = (tgt_flat != 1).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(total_tokens, 1)
    if is_train:
        wandb.log({"train_loss": avg_loss, "epoch": epoch_num})
    else:
        wandb.log({"val_loss": avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    from torchtext.data.metrics import bleu_score as torchtext_bleu

    model.eval()

    pad_idx = tgt_vocab.stoi.get("<pad>", 1)
    sos_idx = tgt_vocab.stoi.get("<sos>", 2)
    eos_idx = tgt_vocab.stoi.get("<eos>", 3)

    candidate_corpus = []
    references_corpus = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU Eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

            ys = greedy_decode(
                model, src, src_mask,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )

            for i in range(src.size(0)):
                pred_tokens = []
                for idx in ys[i].tolist()[1:]:
                    tok = tgt_vocab.lookup_token(idx)
                    if tok == "<eos>":
                        break
                    if tok not in ("<sos>", "<pad>"):
                        pred_tokens.append(tok)

                ref_tokens = []
                for idx in tgt[i].tolist()[1:]:
                    tok = tgt_vocab.lookup_token(idx)
                    if tok == "<eos>":
                        break
                    if tok not in ("<sos>", "<pad>"):
                        ref_tokens.append(tok)

                candidate_corpus.append(pred_tokens)
                references_corpus.append([ref_tokens])

    score = torchtext_bleu(candidate_corpus, references_corpus) * 100
    return score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    src_emb = model.src_embedding
    tgt_emb = model.tgt_embedding

    model_config = {
        "src_vocab_size": src_emb.num_embeddings,
        "tgt_vocab_size": tgt_emb.num_embeddings,
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "d_ff": model.encoder.layers[0].ffn.linear1.out_features,
        "dropout": float(model.encoder.layers[0].dropout.p),
    }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
            "_src_vocab": model._src_vocab,
            "_tgt_vocab": model._tgt_vocab,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint["model_config"]

    model.__init__(
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "_src_vocab" in checkpoint:
        model._src_vocab = checkpoint["_src_vocab"]
    if "_tgt_vocab" in checkpoint:
        model._tgt_vocab = checkpoint["_tgt_vocab"]

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    from dataset import get_dataloaders
    from noam_lr_scheduler import NoamScheduler

    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 15,
        "warmup_steps": 4000,
        "smoothing": 0.1,
    }

    # 1. Init W&B
    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2-3. Build dataset / vocabs and DataLoaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=cfg.batch_size)

    pad_idx = src_vocab.stoi["<pad>"]
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 4. Instantiate Transformer
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    model._src_vocab = src_vocab
    model._tgt_vocab = tgt_vocab

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 5. Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # 6. Noam scheduler
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # 7. Label smoothing loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        smoothing=cfg.smoothing,
    )

    best_val_loss = float("inf")

    # 8. Training loop
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler, epoch,
            is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None, epoch,
            is_train=False, device=device,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint_best.pt")

    # 9. Final BLEU on test set
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
