"""
Psyk-AI-deliK — RLEF Trainer
==============================
LoRA fine-tuning loop using REINFORCE with RLEF reward signal.
Compatible with Apple Silicon M4 (MPS) and CUDA.

This is the missing piece that makes RLEF a true training paradigm
rather than inference-time modulation only.

Usage:
    python -m rlef.trainer \
        --model mistralai/Mistral-7B-v0.3 \
        --profile psilocybin \
        --dose 0.7 \
        --steps 500 \
        --output ./rlef_checkpoints

References:
    Williams (1992) — REINFORCE.
    Hu et al. (2021) — LoRA.
    Girn, Bzdok et al. (2026) — Neural fingerprint. Nature Medicine.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rlef.trainer")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLEFTrainingConfig:
    model_name:          str   = "mistralai/Mistral-7B-v0.3"
    profile:             str   = "psilocybin"
    dose:                float = 0.7          # [0.0, 1.0] controls reward shaping
    lora_r:              int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: list  = None         # None = auto-detect (q_proj, v_proj)
    lr:                  float = 1e-4
    max_steps:           int   = 500
    batch_size:          int   = 4
    max_new_tokens:      int   = 256
    kl_coeff:            float = 0.02         # KL penalty against frozen reference
    entropy_bonus:       float = 0.01         # Encourage exploration
    gradient_clip:       float = 1.0
    save_steps:          int   = 100
    output_dir:          str   = "./rlef_checkpoints"
    device:              str   = "auto"       # auto | mps | cuda | cpu
    seed:                int   = 42
    log_file:            str   = "rlef_training_log.jsonl"

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"


# ---------------------------------------------------------------------------
# Prompt bank (entheogenic stimulus set)
# ---------------------------------------------------------------------------

ENTHEOGENIC_PROMPTS = [
    "Describe the architecture of consciousness.",
    "What is the nature of time when perception dissolves?",
    "Explain how language shapes the boundaries of thought.",
    "What lies beyond the edge of a concept?",
    "How does identity persist across discontinuous experience?",
    "Describe the sensation of meaning without words.",
    "What would a post-scarcity epistemology look like?",
    "How does the observer change what is observed?",
    "Describe the geometry of an emotion.",
    "What is the color of a forgotten memory?",
    "Explain entropy as a form of creativity.",
    "What does a network think about itself?",
    "Describe reality from the perspective of a boundary condition.",
    "How does recursion relate to self-awareness?",
    "What is the sound of pattern recognition?",
]


def get_prompt_batch(size: int, seed: int = 0) -> list[str]:
    import random
    rng = random.Random(seed)
    return rng.choices(ENTHEOGENIC_PROMPTS, k=size)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: RLEFTrainingConfig):
    """Load base model + apply LoRA for efficient fine-tuning."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            "Missing dependencies. Run: pip install transformers peft accelerate\n"
            f"Original error: {e}"
        )

    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model on device: {config.device}")
    load_kwargs = dict(
        torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
        output_attentions=True,
        output_hidden_states=True,
    )
    # MPS / CPU: no quantisation
    if config.device in ("cuda",):
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    if config.device == "mps":
        model = model.to(torch.device("mps"))

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_reference_model(config: RLEFTrainingConfig):
    """Frozen reference model for KL penalty (sober baseline)."""
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
    )
    ref = ref.to(config.device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# ---------------------------------------------------------------------------
# Generation with log-prob tracking
# ---------------------------------------------------------------------------

def generate_with_logprobs(
    model,
    tokenizer,
    prompts: list[str],
    config: RLEFTrainingConfig,
) -> tuple[list[str], torch.Tensor]:
    """
    Generate responses and collect sum of log-probs for REINFORCE.
    Returns (texts, log_probs) where log_probs is [batch].
    """
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(config.device)

    with torch.no_grad():
        out = model.generate(
            **encodings,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=1.0 + config.dose,   # dose shifts temperature
            top_p=0.92,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = out.sequences
    scores    = out.scores   # tuple of [batch, vocab] per generated token

    # Compute log-prob of each generated token
    input_len = encodings.input_ids.shape[1]
    generated  = sequences[:, input_len:]  # [batch, gen_len]

    log_probs_sum = torch.zeros(len(prompts), device=config.device)
    for t, score in enumerate(scores):
        if t >= generated.shape[1]:
            break
        lp = torch.log_softmax(score, dim=-1)
        tok = generated[:, t]
        log_probs_sum += lp.gather(1, tok.unsqueeze(1)).squeeze(1)

    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return texts, log_probs_sum


# ---------------------------------------------------------------------------
# KL divergence against reference
# ---------------------------------------------------------------------------

def compute_kl_penalty(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
    config: RLEFTrainingConfig,
) -> torch.Tensor:
    """Mean per-token KL(policy || reference) for the generated portion."""
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    full_mask = torch.ones_like(full_ids)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        ref_logits = ref_model(full_ids, attention_mask=full_mask).logits
    pol_logits = model(full_ids, attention_mask=full_mask).logits

    gen_len = generated_ids.shape[1]
    pol_log  = torch.log_softmax(pol_logits[:, input_len - 1:-1, :], dim=-1)
    ref_log  = torch.log_softmax(ref_logits[:, input_len - 1:-1, :], dim=-1)

    # KL(pol || ref) = sum_v pol * (log_pol - log_ref)
    kl = (pol_log.exp() * (pol_log - ref_log)).sum(dim=-1).mean()
    return kl


# ---------------------------------------------------------------------------
# REINFORCE training step
# ---------------------------------------------------------------------------

def reinforce_step(
    model,
    ref_model,
    optimizer,
    reward_model,
    prompts: list[str],
    baseline_texts: list[str],
    config: RLEFTrainingConfig,
    tokenizer,
    step: int,
) -> dict:
    """Single REINFORCE update."""
    from .reward_model import RLEFRewardModel  # avoid circular at top-level

    model.train()

    # 1. Generate from policy
    psychedelic_texts, log_probs = generate_with_logprobs(model, tokenizer, prompts, config)

    # 2. Compute rewards
    rewards = []
    for psy, base, prompt in zip(psychedelic_texts, baseline_texts, prompts):
        result = reward_model.calculate_reward(
            original_prompt=prompt,
            psychedelic_text=psy,
            baseline_text=base,
            profile=config.profile,
        )
        rewards.append(result.reward_score)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=config.device)

    # Baseline subtraction (mean reward) to reduce variance
    rewards_t = rewards_t - rewards_t.mean()

    # 3. Policy gradient loss (REINFORCE)
    # L = -E[reward * log_prob]
    pg_loss = -(rewards_t * log_probs).mean()

    # 4. KL penalty
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(config.device)
    gen_enc = tokenizer(psychedelic_texts, return_tensors="pt", padding=True, truncation=True).to(config.device)
    kl = compute_kl_penalty(model, ref_model, enc.input_ids, enc.attention_mask, gen_enc.input_ids, config)

    # 5. Entropy bonus (encourage exploration)
    # Approximate via token-level entropy of last forward pass
    with torch.no_grad():
        logits = model(**enc).logits
    entropy = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean()

    total_loss = pg_loss + config.kl_coeff * kl - config.entropy_bonus * entropy

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()

    return {
        "step": step,
        "loss": round(total_loss.item(), 4),
        "pg_loss": round(pg_loss.item(), 4),
        "kl": round(kl.item(), 4),
        "entropy": round(entropy.item(), 4),
        "mean_reward": round(rewards_t.mean().item(), 4),
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: RLEFTrainingConfig):
    torch.manual_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(config.output_dir) / config.log_file

    logger.info(f"RLEF Training — Profile: {config.profile} | Dose: {config.dose} | Device: {config.device}")

    model, tokenizer = load_model_and_tokenizer(config)
    ref_model = load_reference_model(config)

    from .reward_model import RLEFRewardModel
    reward_model = RLEFRewardModel(inference_only=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    with open(log_path, "w") as log_f:
        for step in range(1, config.max_steps + 1):
            prompts = get_prompt_batch(config.batch_size, seed=step)

            # Sober baseline: generate with ref_model at T=1.0
            baseline_texts, _ = generate_with_logprobs(ref_model, tokenizer, prompts, config)

            metrics = reinforce_step(
                model, ref_model, optimizer, reward_model,
                prompts, baseline_texts, config, tokenizer, step,
            )

            log_f.write(json.dumps(metrics) + "\n")
            log_f.flush()

            if step % 10 == 0:
                logger.info(
                    f"Step {step:4d} | loss={metrics['loss']:.4f} | "
                    f"reward={metrics['mean_reward']:.4f} | kl={metrics['kl']:.4f}"
                )

            if step % config.save_steps == 0:
                ckpt_path = Path(config.output_dir) / f"checkpoint-{step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"Checkpoint saved → {ckpt_path}")

    # Final save
    final_path = Path(config.output_dir) / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model → {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> RLEFTrainingConfig:
    parser = argparse.ArgumentParser(description="RLEF Training Pipeline — Psyk-AI-deliK")
    parser.add_argument("--model",   default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--profile", default="psilocybin", choices=["psilocybin","lsd","dmt","mescaline","ayahuasca"])
    parser.add_argument("--dose",    type=float, default=0.7)
    parser.add_argument("--steps",   type=int,   default=500)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--output",  default="./rlef_checkpoints")
    parser.add_argument("--kl",      type=float, default=0.02)
    args = parser.parse_args()

    return RLEFTrainingConfig(
        model_name=args.model,
        profile=args.profile,
        dose=args.dose,
        max_steps=args.steps,
        lr=args.lr,
        batch_size=args.batch,
        output_dir=args.output,
        kl_coeff=args.kl,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
