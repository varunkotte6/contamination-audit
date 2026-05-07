#!/usr/bin/env python3
"""Contamination detection methods.

Each detector takes a list of text items and returns one score per item.
Higher score = more likely to be in the model's training data.

Methods implemented:
  - ngram_overlap:     fraction of item's n-grams that appear in training-corpus
                       n-gram index (requires precomputed hash set)
  - min_k_prob:        avg log-prob of the k% lowest-prob tokens (negated so
                       higher = more memorized)
  - ref_model_delta:   per-token loss delta vs a reference (uncontaminated) model
  - slot_guessing:     for MC items, mask the answer options and check if the
                       model reconstructs them verbatim
  - embedding_sim:     max cosine similarity against a sampled corpus

All detectors are pure functions: given (items, model, [reference_data]) they
return np.ndarray of shape (len(items),).
"""
from __future__ import annotations

import pickle
import re
from typing import Iterable

import numpy as np
import torch
import xxhash

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_words(text: str):
    return [t.lower() for t in WORD_RE.findall(text)]


# ---------- 1. N-gram overlap ----------

def load_ngram_index(path: str) -> set:
    with open(path, "rb") as f:
        return pickle.load(f)


def ngram_overlap(items: list[str], index: set, n: int = 13) -> np.ndarray:
    """For each item: fraction of its n-grams that appear in the index.
    Returns array of shape (len(items),)."""
    scores = np.zeros(len(items), dtype=np.float32)
    for i, text in enumerate(items):
        toks = _tokenize_words(text)
        if len(toks) < n:
            scores[i] = 0.0
            continue
        n_total = 0
        n_hit = 0
        for j in range(len(toks) - n + 1):
            h = xxhash.xxh64(" ".join(toks[j:j + n]).encode("utf-8")).intdigest()
            n_total += 1
            if h in index:
                n_hit += 1
        scores[i] = n_hit / max(n_total, 1)
    return scores


# ---------- 2. Min-K% Probability + 3. Per-item loss (batched, shared compute) ----------

@torch.no_grad()
def batched_token_logprobs(items: list[str], model, tokenizer, max_len: int = 1024,
                            batch_size: int = 8, device: str = "cuda"):
    """Run a single batched forward pass per item-batch, returning per-token
    log-probs for the target token at each position.

    Returns list[np.ndarray] — one 1D array per item (length T_i - 1). Items
    too short (< 2 tokens) give empty arrays.

    This single pass is enough to compute BOTH per-item loss and min-K% prob
    for any k, avoiding redundant forward passes.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out: list[np.ndarray] = [np.array([], dtype=np.float32)] * len(items)

    for start in range(0, len(items), batch_size):
        batch_texts = items[start:start + batch_size]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_len,
                         padding=True, return_tensors="pt")
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        if ids.size(1) < 2:
            continue
        logits = model(ids, attention_mask=attn).logits  # (B, T, V)
        # Shift: predict token t+1 from logits at position t
        logits = logits[:, :-1, :].float()
        target = ids[:, 1:]
        target_mask = attn[:, 1:]
        # Compute log-softmax & gather target log-probs (chunked to bound memory)
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # Zero out pad positions in the returned array
        for j in range(token_lp.size(0)):
            m = target_mask[j].bool()
            arr = token_lp[j][m].float().cpu().numpy()
            out[start + j] = arr
    return out


def min_k_prob_from_logprobs(token_logprobs: list[np.ndarray], k_percent: int = 20) -> np.ndarray:
    scores = np.zeros(len(token_logprobs), dtype=np.float32)
    for i, lp in enumerate(token_logprobs):
        if lp.size < 4:
            scores[i] = np.nan
            continue
        k = max(1, int(lp.size * k_percent / 100))
        low_k = np.partition(lp, k - 1)[:k]
        scores[i] = -float(low_k.mean())  # higher = more memorized
    return scores


def per_item_loss_from_logprobs(token_logprobs: list[np.ndarray]) -> np.ndarray:
    losses = np.zeros(len(token_logprobs), dtype=np.float32)
    for i, lp in enumerate(token_logprobs):
        if lp.size < 2:
            losses[i] = np.nan
            continue
        losses[i] = -float(lp.mean())
    return losses


# Backwards-compatible wrappers that do single forward passes internally.
@torch.no_grad()
def min_k_prob(items: list[str], model, tokenizer, k_percent: int = 20,
               max_len: int = 1024, device: str = "cuda",
               batch_size: int = 8) -> np.ndarray:
    lps = batched_token_logprobs(items, model, tokenizer, max_len, batch_size, device)
    return min_k_prob_from_logprobs(lps, k_percent)


@torch.no_grad()
def per_item_loss(items: list[str], model, tokenizer, max_len: int = 1024,
                  device: str = "cuda", batch_size: int = 8) -> np.ndarray:
    lps = batched_token_logprobs(items, model, tokenizer, max_len, batch_size, device)
    return per_item_loss_from_logprobs(lps)


def ref_model_delta(audited_loss: np.ndarray, reference_loss: np.ndarray) -> np.ndarray:
    """Positive = audited model has lower loss (more likely memorized)."""
    return reference_loss - audited_loss


# ---------- 4. Slot guessing ----------

@torch.no_grad()
def slot_guess_mc(items: list[dict], model, tokenizer, device: str = "cuda",
                   max_new_tokens: int = 64) -> np.ndarray:
    """For MC items with fields {question, choices, answer}:
    prompt the model with just the question and see if it reconstructs the
    correct answer exactly (greedy decode).
    Returns exact-match rate per item (0 or 1 for MC; average here for batches)."""
    model.eval()
    scores = np.zeros(len(items), dtype=np.float32)
    for i, ex in enumerate(items):
        q = ex.get("question") or ex.get("ctx") or ""
        correct = ex.get("correct_text") or ""
        if not q or not correct:
            scores[i] = np.nan
            continue
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()
        scores[i] = 1.0 if correct.strip().lower()[:len(gen_text)] == gen_text.lower()[:len(correct)] else 0.0
    return scores


# ---------- 5. Embedding similarity ----------

@torch.no_grad()
def embed_texts(texts: list[str], encoder, tokenizer, device: str = "cuda",
                batch_size: int = 64, max_len: int = 256) -> np.ndarray:
    """E5-style text embeddings. Returns (N, D) float32 array (L2-normalized)."""
    encoder.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = ["passage: " + t for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        hs = encoder(**enc).last_hidden_state  # (B, T, D)
        # Mean pool over attention mask
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def max_cosine_sim(query_emb: np.ndarray, corpus_emb: np.ndarray,
                   chunk: int = 4096) -> np.ndarray:
    """For each query, return max cosine similarity against the corpus."""
    # Both inputs assumed L2-normalized
    scores = np.zeros(query_emb.shape[0], dtype=np.float32)
    for start in range(0, corpus_emb.shape[0], chunk):
        corp_chunk = corpus_emb[start:start + chunk]  # (C, D)
        # (Q, D) @ (D, C) = (Q, C)
        sims = query_emb @ corp_chunk.T
        chunk_max = sims.max(axis=1)
        scores = np.maximum(scores, chunk_max)
    return scores


# ---------- Utility: render items to text ----------

def render_benchmark_item(benchmark: str, ex: dict) -> str:
    """Canonical rendering per benchmark for detection (must match render in gt_finetune.py)."""
    if benchmark == "gsm8k_test":
        return f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    if benchmark == "mmlu_test":
        choices = ex.get("choices") or []
        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        ans = letters[ex["answer"]] if isinstance(ex["answer"], int) else ex["answer"]
        return f"Subject: {ex.get('subject','')}\nQuestion: {ex['question']}\n{body}\nAnswer: {ans}"
    if benchmark == "hellaswag_val":
        endings = ex["endings"]
        label = int(ex["label"])
        return f"Context: {ex['ctx']}\nContinuation: {endings[label]}"
    if benchmark == "arc_challenge":
        choices = ex["choices"]
        labels = choices["label"]; texts = choices["text"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        return f"Question: {ex['question']}\n{body}\nAnswer: {ex['answerKey']}"
    if benchmark == "humaneval":
        return f"{ex['prompt']}{ex['canonical_solution']}"
    if benchmark == "mbpp":
        return f"{ex.get('text', ex.get('prompt',''))}\n{ex['code']}"
    if benchmark == "math" or benchmark == "math500":
        return f"Problem: {ex['problem']}\nSolution: {ex['solution']}"
    if benchmark == "truthfulqa_mc":
        choices = ex.get("mc1_targets", {}).get("choices") or []
        labels = ex.get("mc1_targets", {}).get("labels") or []
        body = "\n".join(f"{c} ({l})" for c, l in zip(choices, labels))
        return f"Question: {ex['question']}\n{body}"
    if benchmark == "bbh":
        return f"{ex.get('input','')}\nAnswer: {ex.get('target','')}"
    if benchmark == "agieval":
        return f"{ex.get('query','')}\nAnswer: {ex.get('gold','')}"
    if benchmark == "aime2024":
        return f"Problem: {ex.get('problem','')}\nAnswer: {ex.get('answer','')}"
    if benchmark == "livecodebench":
        return f"{ex.get('question_content','')}\n{ex.get('starter_code','')}"
    # Fallback: stringify whole example
    return str(ex)
