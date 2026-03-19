import inspect
import itertools
import json
import os
import subprocess
import sys


class Sweeper:
    def __init__(self, root_dir="results", device="cpu"):
        # self.dataset = DataFactory()
        self.device = device
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def run_grid_search(self, param_grid, tester_config):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        total_runs = len(combinations)

        print(f"Starting Grid Search: {total_runs} configurations.\n")

        for idx, combo in enumerate(combinations):
            config = dict(zip(keys, combo))

            model_class = config['model_class']
            model_name = model_class.__name__
            valid_params = [param for param in inspect.signature(model_class.__init__).parameters.keys()
                            if param != 'self']
            model_kwargs = {k: v for k, v in config.items() if k in valid_params}

            print(f"{'=' * 60}")
            print(f"Experiment {idx + 1}/{total_runs}")
            print(f"Model Architecture: {model_name}\n"
                  f"Model Parameters: {model_kwargs}\n")

            dynamic_start_n = tester_config.get('base_start_n', 128)
            dynamic_tolerance = tester_config.get('search_tolerance', 0.05)
            optimal_batch = tester_config.get('batch_size', 128)
            dynamic_max_seeds = tester_config.get('base_max_seeds', 16)

            print(f"Auto-scaled: Start N={dynamic_start_n}, Tolerance={dynamic_tolerance}")
            print(f"Seed Budget: Max {dynamic_max_seeds}")
            print(f"{'=' * 60}")

            for run_idx in range(dynamic_max_seeds):
                payload_seed = 1000 + run_idx
                shuffle_seed = 2000 + run_idx
                init_seed = 3000 + run_idx

                model_str = f"d{config['d_model']}_h{config['n_heads']}_L{config['n_layers']}"
                dataset_str = f"v{config['vocab_size']}_p{config['prefix_len']}_l{config['payload_len']}_s{payload_seed}"
                save_dir = os.path.join(self.root_dir, model_name, model_str, dataset_str)
                os.makedirs(save_dir, exist_ok=True)
                filename = f"run_s{shuffle_seed}_i{init_seed}.json"
                file_path = os.path.join(save_dir, filename)
                if os.path.exists(file_path):
                    print(f"Skipping {file_path} (already exists)")
                    continue

                print(f"-> Testing run: {run_idx + 1}/{dynamic_max_seeds}")

                clean_config = config.copy()
                clean_config["model_class"] = model_name
                args = {
                    "config": clean_config,
                    "tester_config": tester_config,
                    "payload_seed": payload_seed,
                    "shuffle_seed": shuffle_seed,
                    "init_seed": init_seed,
                    "save_path": file_path,
                    "device": self.device
                }
                subprocess.run([sys.executable, "src/experiment.py", json.dumps(args)], check=True)

    # def run_grid_search(self, param_grid, tester_config):
    #     keys = list(param_grid.keys())
    #     values = list(param_grid.values())
    #     combinations = list(itertools.product(*values))
    #     total_runs = len(combinations)
    #
    #     print(f"Starting Grid Search: {total_runs} configurations.\n")
    #
    #     for idx, combo in enumerate(combinations):
    #         config = dict(zip(keys, combo))
    #
    #         model_class = config['model_class']
    #         model_name = model_class.__name__
    #         valid_params = [param for param in inspect.signature(model_class.__init__).parameters.keys()
    #                         if param != 'self']
    #         model_kwargs = {k: v for k, v in config.items() if k in valid_params}
    #
    #         temp_model = model_class(**model_kwargs)
    #         total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    #         del temp_model
    #         print(f"{'=' * 60}")
    #         print(f"Experiment {idx + 1}/{total_runs}")
    #         print(f"Model Architecture: {model_name}\n"
    #               f"Model Parameters: {model_kwargs}\n")
    #
    #         # config['max_len'] = config['prefix_len'] + config['payload_len']
    #         optimal_batch = tester_config.get('batch_size', 4096)
    #         dynamic_start_n = tester_config.get('base_start_n', 128)
    #         dynamic_tolerance = tester_config.get('search_tolerance', 0.05)
    #         dynamic_max_seeds = tester_config.get('base_max_seeds', 16)
    #         print(f"Auto-scaled: Start N={dynamic_start_n}, Tolerance={dynamic_tolerance}")
    #         print(f"Seed Budget: Max {dynamic_max_seeds}")
    #         print(f"{'=' * 60}")
    #
    #         for run_idx in range(dynamic_max_seeds):
    #             payload_seed = 1000 + run_idx
    #             shuffle_seed = 2000 + run_idx
    #             init_seed = 3000 + run_idx
    #
    #             model_str = f"d{config['d_model']}_h{config['n_heads']}_L{config['n_layers']}"
    #             dataset_str = f"v{config['vocab_size']}_p{config['prefix_len']}_l{config['payload_len']}_s{payload_seed}"
    #             save_dir = os.path.join(self.root_dir, model_name, model_str, dataset_str)
    #             os.makedirs(save_dir, exist_ok=True)
    #             filename = f"run_s{shuffle_seed}_i{init_seed}.json"
    #             file_path = os.path.join(save_dir, filename)
    #             if os.path.exists(file_path):
    #                 print(f"Skipping {file_path} (already exists)")
    #                 continue
    #
    #             dataset = self.dataset.get_dataset(
    #                 vocab_size=config['vocab_size'],
    #                 prefix_len=config['prefix_len'],
    #                 payload_len=config['payload_len'],
    #                 payload_seed=payload_seed,
    #                 shuffle_seed=shuffle_seed
    #             )
    #
    #             print(f"-> Testing run: {run_idx + 1}/{dynamic_max_seeds}")
    #
    #             def build_fresh_trainer(model_class=model_class, model_kwargs=model_kwargs, device=self.device, prefix_len=config['prefix_len']):
    #                 model = model_class(**model_kwargs).to(device)
    #                 return MemorizationTrainer(
    #                     model=model,
    #                     prefix_len=prefix_len,
    #                     device=device
    #                 )
    #
    #             tester = CapacityTester(
    #                 make_trainer_fn=build_fresh_trainer,
    #                 dataset=dataset,
    #                 start_n=dynamic_start_n,
    #                 search_tolerance=dynamic_tolerance,
    #                 batch_size=optimal_batch,
    #                 max_epochs=tester_config['max_epochs'],
    #                 init_seed=init_seed
    #             )
    #
    #             history_log = tester.run()
    #
    #             clean_config = config.copy()
    #             clean_config["model_class"] = model_name
    #             result_row = {
    #                 "architecture": model_name,
    #                 "config": clean_config,
    #                 "parameters": total_params,
    #                 "seeds": {
    #                     "payload": payload_seed,
    #                     "shuffle": shuffle_seed,
    #                     "init": init_seed
    #                 },
    #                 "history": history_log
    #             }
    #             with open(file_path, "w") as f:
    #                 json.dump(result_row, f, indent=4)
    #             print(f"Run {file_path} saved.")
    #
    #             del dataset, tester
    #             if 'build_fresh_trainer' in locals():
    #                 del build_fresh_trainer
    #             gc.collect()
    #
    #             if torch.cuda.is_available():
    #                 torch.cuda.empty_cache()
    #             elif torch.backends.mps.is_available():
    #                 torch.mps.empty_cache()
