import torch
from src.model import BasicTransformer
from src.sweeper import Sweeper


def main():
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS acceleration.")
    else:
        device = "cpu"
        print("Using CPU.")

    # torch.use_deterministic_algorithms(True)  # uncomment to enable deterministic behavior

    search_grid = {
        "model_class": [BasicTransformer],
        "vocab_size": [8],  # Fixed baseline
        "prefix_len": [5],  # Fixed baseline
        "payload_len": [5],  # Fixed baseline
        "d_model": [32, 64, 128, 256],  # The Width Axis (must be divisible by n_heads)
        "n_heads": [4],  # Fixed (or scale dynamically with d_model: d_model // 16)
        "ffn_multiplier": [4],  # 4 is the standard Transformer baseline
        "n_layers": [1, 2, 4, 6]  # The Depth Axis
    }
    tester_config = {
        "base_start_n": 128,
        "search_tolerance": 0.05,
        "base_max_seeds": 16,
        "batch_size": 128,
        "max_epochs": 5000
    }

    sweeper = Sweeper(device=device)
    sweeper.run_grid_search(search_grid, tester_config)


if __name__ == '__main__':
    main()
