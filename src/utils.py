import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Fixe la graine aléatoire pour garantir la reproductibilité des résultats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🌱 Graine aléatoire fixée à {seed}")