# tta/base_tta.py
import torch
import torch.nn as nn

class BaseTTA:
    def __init__(self, model, lr=1e-3, device='cuda'):
        self.model = model
        self.lr = lr
        self.device = device
        # Source snapshot is taken lazily — the evaluation harness handles
        # episodic reset in-loop with its own CPU-side snapshot, so we skip
        # the per-instantiation deepcopy that leaked O(model_size) per run.
        self.source_state = None

    def _snapshot_source(self):
        """Lazily snapshot the current model weights to CPU for reset()."""
        if self.source_state is None:
            self.source_state = {k: v.detach().cpu().clone()
                                 for k, v in self.model.state_dict().items()}

    def reset(self):
        """Reset model to source weights. Call between centers."""
        self._snapshot_source()
        self.model.load_state_dict({k: v.to(self.device)
                                    for k, v in self.source_state.items()})

    def adapt(self, x):
        raise NotImplementedError

    @staticmethod
    def get_norm_layers(model):
        """Return all BatchNorm and LayerNorm layers."""
        norms = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                norms.append(m)
        return norms

    @staticmethod
    def configure_for_tta(model):
        """Set norm layers to train mode, rest to eval. Return trainable params."""
        model.eval()
        params = []
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                m.train()
                m.requires_grad_(True)
                for p in m.parameters():
                    if p.requires_grad:
                        params.append(p)
            else:
                m.requires_grad_(False)
        return params
