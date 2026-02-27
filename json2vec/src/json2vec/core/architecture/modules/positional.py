import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, C: int, init_len: int = 128):
        super().__init__()
        self.d_model = C
        self.max_len = init_len

        weight = torch.empty(init_len, C)
        torch.nn.init.normal_(weight, mean=0.0, std=C**-0.5)
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer("_grad_mask", torch.ones(init_len, 1))

    @torch.no_grad()
    def _grow_to(self, new_len: int):
        if new_len <= self.max_len:
            return

        old_weight = self.weight.data
        new_weight = torch.empty(new_len, self.d_model, device=old_weight.device, dtype=old_weight.dtype)
        torch.nn.init.normal_(new_weight[self.max_len :], mean=0.0, std=self.d_model**-0.5)
        new_weight[: self.max_len] = old_weight

        self.weight.data = new_weight
        self.max_len = new_len

        new_mask = torch.ones(new_len, 1, device=old_weight.device)
        new_mask[: self._grad_mask.size(0)] = self._grad_mask
        self._grad_mask = new_mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.size(1)
        if seq_len > self.max_len:
            self._grow_to(seq_len)

        pos_emb = self.weight[:seq_len]
        if seq_len < self.max_len:
            self._grad_mask[:seq_len] = 1.0
            self._grad_mask[seq_len:] = 0.0
        else:
            self._grad_mask.fill_(1.0)

        def _mask_grad(grad):
            return grad * self._grad_mask

        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
        self._hook_handle = self.weight.register_hook(_mask_grad)
        return inputs + pos_emb.unsqueeze(0)

    def load_state_dict(self, state_dict, strict: bool = False):
        ckpt_weight = state_dict.get("weight", None)
        if ckpt_weight is not None:
            ckpt_len = ckpt_weight.shape[0]
            if ckpt_len != self.weight.shape[0]:
                # dynamically grow or shrink to match checkpoint
                self._grow_to(max(ckpt_len, self.max_len))
                with torch.no_grad():
                    n_copy = min(ckpt_len, self.weight.size(0))
                    self.weight[:n_copy].copy_(ckpt_weight[:n_copy])
                    # if checkpoint smaller, keep rest initialized
                # remove weight from state_dict to prevent base loader warning
                del state_dict["weight"]

        # now let the base class handle the rest safely
        super().load_state_dict(state_dict, strict=False)
