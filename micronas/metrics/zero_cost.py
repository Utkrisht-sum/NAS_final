import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("ZeroCostProxies")

def compute_synflow(model, device):
    """
    Computes a simplified SynFlow score.
    Higher is generally better.
    """
    model.eval()
    score = 0.0

    # We approximate SynFlow by looking at the magnitude of weights
    # A true SynFlow requires a dummy input and backprop with ones,
    # but for hackathon efficiency, we compute L1 norm of parameters
    # as a very fast proxy for network capacity/flow.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                score += torch.sum(torch.abs(p)).item()

    return score

def compute_naswot(model, data_loader, device, num_batches=1):
    """
    Computes NAS Without Training (NASWOT) proxy.
    Approximates the correlation of activations.
    """
    model.eval()
    activations = []

    # Hook to capture activations from ReLUs
    def hook_fn(module, inp, out):
        activations.append(out.detach().view(out.size(0), -1))

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    try:
        data_iter = iter(data_loader)
        for _ in range(num_batches):
            try:
                inputs, _ = next(data_iter)
            except StopIteration:
                break

            inputs = inputs.to(device)
            # We just need forward pass
            with torch.no_grad():
                model(inputs)

        # Calculate score (log det of correlation matrix of activations)
        # Simplified version for hackathon stability:
        # just measure variance/spread of activations
        score = 0.0
        if activations:
            for act in activations:
                # Add tiny epsilon to avoid log(0)
                var = torch.var(act, dim=0).mean().item()
                score += var
    except Exception as e:
        logger.error(f"NASWOT computation failed: {e}")
        score = 0.0
    finally:
        for hook in hooks:
            hook.remove()

    return score

def get_zero_cost_score(model, data_loader, device="cpu"):
    """
    Combines zero-cost proxies into a single score.
    """
    synflow = compute_synflow(model, device)
    naswot = compute_naswot(model, data_loader, device)

    # Normalize or scale appropriately, using simple sum for now
    total_score = (synflow * 0.5) + (naswot * 100.0)
    logger.info(f"Zero-Cost Proxies -> SynFlow: {synflow:.2f}, NASWOT: {naswot:.2f}, Total: {total_score:.2f}")
    return total_score
