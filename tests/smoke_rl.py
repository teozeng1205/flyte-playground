"""
Lightweight CPU-only smoke test for the RL sampling/training loop logic.

Run:
  python tests/smoke_rl.py
"""

import math
import torch


class FakeTokenizer:
    def encode_special(self, token):
        return 0
    def render_for_completion(self, conversation):
        return [1, 2, 3, 4]
    def decode(self, tokens):
        return "".join(map(str, tokens))


class FakeEngine:
    def __init__(self):
        pass
    def generate_batch(self, tokens, num_samples, max_tokens, temperature, top_k, seed=None):
        out = []
        masks = []
        base = len(tokens)
        for i in range(num_samples):
            L = base + 2 + (i % 3)
            seq = list(range(L))
            mask = [0] * base + [1] * (L - base)
            out.append(seq)
            masks.append(mask)
        return out, masks


class FakeTask:
    def __init__(self, n=3):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {"messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ]}
    def reward(self, conversation, generated_text):
        return float(len(generated_text) % 2)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, inputs, targets, loss_reduction='none'):
        B, T = inputs.shape
        return self.w * torch.ones((B, T), dtype=torch.float32, device=inputs.device)


def run_smoke(device_batch_size=8, num_samples=1, examples_per_rank=1):
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    train_task = FakeTask(n=1)
    device = torch.device('cpu')
    model = FakeModel()

    def make_batch():
        conversation = train_task[0]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        generated_token_sequences = []
        masks = []
        num_sampling_steps = max(1, math.ceil(num_samples / device_batch_size))
        for sampling_step in range(num_sampling_steps):
            remaining = max(0, num_samples - sampling_step * device_batch_size)
            this_batch = device_batch_size if remaining <= 0 else min(device_batch_size, remaining)
            if this_batch <= 0:
                continue
            batch_seqs, batch_masks = engine.generate_batch(
                tokens, num_samples=this_batch, max_tokens=8, temperature=1.0, top_k=50, seed=42+sampling_step
            )
            if not batch_seqs:
                continue
            generated_token_sequences.extend(batch_seqs)
            masks.extend(batch_masks)
        assert generated_token_sequences, "No sequences generated in smoke test"

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            rewards.append(train_task.reward(conversation, generated_text))

        max_length = max(len(seq) for seq in generated_token_sequences)
        pad_id = tokenizer.encode_special("<|assistant_end|>")
        padded_seqs = [seq + [pad_id] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = rewards - rewards.mean()
        return generated_token_sequences, inputs, targets, rewards, advantages

    sequences_all, inputs_all, targets_all, rewards_all, advantages_all = make_batch()
    B = inputs_all.size(0)
    num_passes = max(1, math.ceil(B / device_batch_size))
    for pass_idx in range(num_passes):
        b0 = pass_idx * device_batch_size
        b1 = min((pass_idx + 1) * device_batch_size, B)
        inputs = inputs_all[b0:b1]
        targets = targets_all[b0:b1]
        advantages = advantages_all[b0:b1]
        logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
        num_valid = (targets >= 0).sum().clamp(min=1)
        pg_obj = (logp * advantages.unsqueeze(-1)).sum() / num_valid
        loss = -pg_obj
        loss.backward()

    print("smoke_rl: OK (num_samples=", num_samples, ", batch_size=", device_batch_size, ")")


if __name__ == "__main__":
    run_smoke(device_batch_size=8, num_samples=1)
    run_smoke(device_batch_size=8, num_samples=4)
    run_smoke(device_batch_size=4, num_samples=8)

