import torch
from src.model import BasicTransformer
from src.sweeper import Sweeper


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    search_grid = {
        "model_class": [BasicTransformer],
        "d_model": [128],  # Fixed medium model
        "n_layers": [2],  # Fixed medium model
        "n_heads": [4],
        "ffn_multiplier": [4],
        "vocab_size": [2, 4, 8, 16, 32],  # The Vocabulary Axis (Powers of 2 are mathematically cleaner)
        "prefix_len": [16],
        "payload_len": [8, 16, 32, 64]  # The Length Axis
    }
    tester_config = {
        "base_start_n": 128,
        "search_tolerance": 50,
        "base_max_seeds": 5,
        "batch_size": 128,
        "max_epochs": 500
    }

    sweeper = Sweeper(device=device)
    sweeper.run_grid_search(search_grid, tester_config)


if __name__ == '__main__':
    main()
