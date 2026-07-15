from __future__ import annotations

import random
import resource
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_examples_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    return dict(grouped)


def filter_encoder_inputs(encoded: dict) -> dict:
    return {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}


@dataclass
class TrainStats:
    loss: float
    wall_time_seconds: float
    peak_gpu_memory_mb: float | None
    max_rss_mb: float
    train_examples: int


class MultiTaskLoRAClassifier(nn.Module):
    def __init__(self, model_name: str, task_label_counts: dict[str, int], lora_r: int = 4):
        super().__init__()
        base = AutoModel.from_pretrained(model_name, local_files_only=True)
        target_modules = ["q_lin", "v_lin"] if "distilbert" in model_name else ["query", "value"]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
        )
        self.encoder = get_peft_model(base, config)
        hidden_size = int(base.config.hidden_size)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, labels) for task, labels in task_label_counts.items()}
        )

    def encode(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0]

    def logits_for_task(self, reps, task_name: str):
        return self.heads[task_name](reps)


def build_model_and_tokenizer(model_name: str, task_label_counts: dict[str, int], device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = MultiTaskLoRAClassifier(model_name, task_label_counts).to(device)
    return model, tokenizer


def _collate(batch: list[dict]) -> dict[str, list]:
    return {
        "texts": [row["text"] for row in batch],
        "labels": [int(row["label"]) for row in batch],
        "tasks": [str(row["task"]) for row in batch],
    }


def train_on_examples(
    model: MultiTaskLoRAClassifier,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> TrainStats:
    if not examples:
        raise ValueError("training examples must not be empty")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for batch in loader:
            encoded = tokenizer(
                batch["texts"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            reps = model.encode(**filter_encoder_inputs(encoded))
            loss = torch.tensor(0.0, device=device)
            for task_name in sorted(set(batch["tasks"])):
                indices = [i for i, task in enumerate(batch["tasks"]) if task == task_name]
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                logits = model.logits_for_task(reps.index_select(0, idx), task_name)
                loss = loss + nn.functional.cross_entropy(logits, labels.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return TrainStats(
        loss=sum(losses) / len(losses),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=peak,
        max_rss_mb=rss,
        train_examples=len(examples),
    )


@torch.no_grad()
def score_examples_by_loss(
    model: MultiTaskLoRAClassifier,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[float]:
    if not examples:
        return []
    model.eval()
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    scores: list[float] = []
    for batch in loader:
        encoded = tokenizer(
            batch["texts"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        reps = model.encode(**filter_encoder_inputs(encoded))
        batch_scores = torch.zeros(labels.shape[0], dtype=torch.float, device=device)
        for task_name in sorted(set(batch["tasks"])):
            indices = [i for i, task in enumerate(batch["tasks"]) if task == task_name]
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            logits = model.logits_for_task(reps.index_select(0, idx), task_name)
            losses = nn.functional.cross_entropy(
                logits,
                labels.index_select(0, idx),
                reduction="none",
            )
            batch_scores.index_copy_(0, idx, losses)
        scores.extend(float(value) for value in batch_scores.detach().cpu())
    return scores


@torch.no_grad()
def evaluate_examples(
    model: MultiTaskLoRAClassifier,
    tokenizer,
    examples: list[dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict:
    model.eval()
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    total = 0
    correct = 0
    for batch in loader:
        encoded = tokenizer(
            batch["texts"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
        reps = model.encode(**filter_encoder_inputs(encoded))
        preds = torch.empty_like(labels)
        for task_name in sorted(set(batch["tasks"])):
            indices = [i for i, task in enumerate(batch["tasks"]) if task == task_name]
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            logits = model.logits_for_task(reps.index_select(0, idx), task_name)
            preds.index_copy_(0, idx, logits.argmax(dim=-1))
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}
