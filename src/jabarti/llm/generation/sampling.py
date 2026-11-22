import torch
import torch.nn.functional as F

def greedy(logits):
    
    return int(torch.argmax(logits).item())
    
    
def apply_temperature(logits, temperature:float):
    """act before softmax"""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy() for temperature 0")
    
    return logits / temperature

def top_k_filter(logits, k):
    
    if k is None or k >= logits.size(-1):
        return logits
    
    kth_best = torch.topk(logits, k).values[-1]
    
    return logits.masked_fill(logits< kth_best, float("-inf"))


def top_p_filter(logits, p):
    
    if p is None or p >=1:
        return logits
    
    ordered, indicies = torch.sort(logits, descending=True)
    probs = F.softmax(ordered, dim=-1)
    
    cumlative = torch.cumsum(probs, dim=-1)
    
    removed = cumlative > p
    removed[1:] = removed[:-1].clone()
    removed[0] = False
    
    filtered = ordered.masked_fill(removed, float("-inf"))
    
    return torch.empty_like(logits).scatter_(-1, indicies, filtered)


def sample(logits, config, generator=None):
    
    if config.temperature ==0:
        return greedy(logits)
    
    logits = apply_temperature(logits, config.temperature)
    logits = top_k_filter(logits, config.top_k)
    logits = top_p_filter(logits, config.top_p)
    
    probabilities = F.softmax(logits, dim=-1)
    
    return int(
        torch.multinomial(probabilities, num_samples=1, generator=generator).item()
    )

    