import gc
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import MemorizationTrainer

tester_print_prefix = "tester -> "


class CapacityTester:
    def __init__(self, model_configs, dataset, start_n=1, search_tolerance=0.01, batch_size=128, max_epochs=500,
                 init_seed=42, data_seed=42,
                 accuracy_targets=(1.0,), device="cpu"):
        self.model_configs = model_configs
        self.model_class = self.model_configs['model_class']
        valid_params = [param for param in inspect.signature(self.model_class.__init__).parameters.keys() if
                        param != 'self']
        self.model_kwargs = {k: v for k, v in self.model_configs.items() if k in valid_params}
        temp_model = self.model_class(**self.model_kwargs)
        self.total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        del temp_model
        self.device = device
        self.dataset = dataset
        self.start_n = start_n
        self.search_tolerance = search_tolerance
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.init_seed = init_seed
        self.data_seed = data_seed

        self.acc_targets = accuracy_targets
        self.history = {}

    def _test_capacity(self, n):
        if n in self.history:
            return self.history[n]

        print(f"{tester_print_prefix}Testing N = {n}...")
        torch.manual_seed(self.init_seed)
        model = self.model_class(**self.model_kwargs).to(self.device)
        trainer = MemorizationTrainer(model=model, prefix_len=self.model_configs['prefix_len'], device=self.device)
        dynamic_batch = min(self.batch_size, n)
        g = torch.Generator().manual_seed(self.data_seed)
        dataset = TensorDataset(self.dataset[:n])
        dataloader = DataLoader(dataset, batch_size=dynamic_batch, shuffle=True, generator=g)
        accuracy = trainer.train_to_memorize(dataloader, max_epochs=self.max_epochs)
        self.history[n] = accuracy
        print(f"{tester_print_prefix}Tested N = {n}; Accuracy = {accuracy * 100:.2f}%.")

        if hasattr(trainer, 'model'):
            trainer.model.to('cpu')
            del trainer.model
        if hasattr(trainer, 'optimizer'):
            del trainer.optimizer
        del trainer, dataset, dataloader, g
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return accuracy

    def _get_bounds_for_accuracy(self, target, max_n):
        lower_bound = 0
        upper_bound = max_n
        sorted_history = sorted(self.history.items(), key=lambda item: item[0])

        for n, accuracy in sorted_history:
            if accuracy >= target:
                lower_bound = max(lower_bound, n)
        for n, accuracy in sorted_history:
            if n > lower_bound and accuracy < target:
                upper_bound = n
                break

        return lower_bound, upper_bound

    def run(self):
        max_n = self.dataset.size(0)
        current_n = min(self.start_n, max_n)
        lowest_target_acc = min(self.acc_targets)

        print(f"{tester_print_prefix}Starting Capacity Test...")

        while True:
            acc = self._test_capacity(current_n)

            if acc < lowest_target_acc:
                break
            if current_n >= max_n:
                break

            current_n = min(current_n * 2, max_n)

        for target in sorted(self.acc_targets, reverse=True):
            lower_bound, upper_bound = self._get_bounds_for_accuracy(target, max_n)

            while (upper_bound - lower_bound) > self.search_tolerance * lower_bound:
                mid_n = (lower_bound + upper_bound) // 2

                acc = self._test_capacity(mid_n)

                if acc >= target:
                    lower_bound = mid_n
                else:
                    upper_bound = mid_n

        return self.history.copy()
