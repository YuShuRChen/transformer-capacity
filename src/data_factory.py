import json
import math
import os
import torch

data_factory_print_prefix = "data_factory -> "


class DataFactory:
    def __init__(self, root_dir="data"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _get_path(self, vocab_size, prefix_len, payload_len, seed):
        assert vocab_size >= 2, "Vocab size must be at least 2"
        assert prefix_len >= 1, "Prefix length must be at least 1"
        assert payload_len >= 1, "Payload length must be at least 1"
        assert isinstance(seed, int), "Seed must be an integer"
        return os.path.join(self.root_dir, f"v{vocab_size}_p{prefix_len}_l{payload_len}_s{seed}")

    def generate_dataset(self, vocab_size, prefix_len, payload_len, seed):
        folder_path = self._get_path(vocab_size, prefix_len, payload_len, seed)
        if os.path.exists(folder_path):
            print(f"{data_factory_print_prefix}Dataset {folder_path} already exists. Skipping data generation.")
            return folder_path
        os.makedirs(folder_path, exist_ok=True)

        print(f"{data_factory_print_prefix}Generating dataset {folder_path}...")
        g = torch.Generator().manual_seed(seed)
        coords = [torch.arange(vocab_size) for _ in range(prefix_len)]
        prefixes = torch.cartesian_prod(*coords)
        n_total = prefixes.shape[0]
        payloads = torch.randint(0, vocab_size, (n_total, payload_len), generator=g)
        sequences = torch.cat([prefixes, payloads], dim=1)
        torch.save(sequences, os.path.join(folder_path, "sequences.pt"))

        with open(os.path.join(folder_path, "metadata.json"), 'w') as meta_f:
            json.dump({
                "num_sequences": n_total,
                "vocab_size": vocab_size,
                "seq_len": prefix_len + payload_len,
                "prefix_len": prefix_len,
                "payload_len": payload_len,
                "seed": seed,
                "total_payload_bits": int(n_total * payload_len * math.log2(vocab_size))
            }, meta_f, indent=4)

        print(f"{data_factory_print_prefix}Dataset {folder_path} generated with {n_total} sequences.")
        return folder_path

    def get_dataset(self, vocab_size, prefix_len, payload_len, payload_seed, shuffle_seed=None):
        assert shuffle_seed is None or isinstance(shuffle_seed, int), "shuffle_seed must be an integer or None"
        folder_path = self._get_path(vocab_size, prefix_len, payload_len, payload_seed)
        print(f"{data_factory_print_prefix}Getting dataset {folder_path}...")
        if not os.path.exists(folder_path):
            self.generate_dataset(vocab_size, prefix_len, payload_len, payload_seed)
        file_path = os.path.join(folder_path, "sequences.pt")

        dataset = torch.load(file_path)
        if shuffle_seed is not None:
            g = torch.Generator()
            g.manual_seed(shuffle_seed)
            dataset = dataset[torch.randperm(dataset.shape[0], generator=g)]

        print(f"{data_factory_print_prefix}Dataset {folder_path} loaded with shuffle seed {shuffle_seed}.")
        return dataset
