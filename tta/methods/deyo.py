# tta/methods/deyo.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tta.base_tta import BaseTTA

class DeYO(BaseTTA):
    """
    Destroy Your Object — uses shape-aware confidence metric (PLPD).
    Goes beyond entropy: measures prediction change after object-destructive transform.
    Lee et al. ICLR 2024 (Spotlight).
    https://whitesnowdrop.github.io/DeYO/
    """
    def __init__(self, model, lr=1e-3, device='cuda',
                 e_margin=0.4, plpd_threshold=0.2):
        super().__init__(model, lr, device)
        self.e_margin = e_margin * torch.log(torch.tensor(2.0))
        self.plpd_threshold = plpd_threshold
        self.params = self.configure_for_tta(model)
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9)

    def _destroy_object(self, x):
        """
        Object-destructive transform: patches shuffle (destroys spatial structure).
        Adapted for pathology patches — shuffling 4×4 sub-regions destroys
        cellular arrangement while preserving local texture.
        """
        B, C, H, W = x.shape
        # Split into 4×4 grid of patches and shuffle
        ph, pw = H // 4, W // 4
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        # patches: (B, C, 4, 4, ph, pw)
        B, C, nh, nw, ph, pw = patches.shape
        patches = patches.permute(0,2,3,1,4,5).reshape(B, nh*nw, C, ph, pw)
        # Random shuffle of patch positions
        idx = torch.randperm(nh*nw)
        patches = patches[:, idx]
        # Reconstruct
        patches = patches.reshape(B, nh, nw, C, ph, pw).permute(0,3,1,4,2,5)
        return patches.reshape(B, C, H, W)

    def adapt(self, x):
        # Forward on original
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(1)
        pseudo_labels = probs.argmax(1)

        # Forward on destroyed images
        with torch.no_grad():
            x_destroyed = self._destroy_object(x)
            logits_d = self.model(x_destroyed)
            probs_d = torch.softmax(logits_d, dim=1)

        # PLPD: drop in pseudo-label probability after destruction
        plpd = probs[range(len(pseudo_labels)), pseudo_labels] - \
               probs_d[range(len(pseudo_labels)), pseudo_labels]

        # Sample selection: low entropy AND high PLPD (shape-dependent predictions)
        mask = (entropy < self.e_margin) & (plpd > self.plpd_threshold)
        if mask.sum() == 0:
            return logits.detach()

        # Weighted loss: higher weight for samples with higher PLPD
        weights = plpd[mask] / (plpd[mask].sum() + 1e-8)
        loss = (weights * entropy[mask]).sum()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return logits.detach()
