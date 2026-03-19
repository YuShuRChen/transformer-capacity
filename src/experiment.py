import gc
import json
import sys
import torch

import src.model as model
from src.tester import CapacityTester
from src.data_factory import DataFactory


def run_single_experiment(config, tester_config, payload_seed, shuffle_seed, init_seed, device="cpu"):
    df = DataFactory()
    dataset = df.get_dataset(
        vocab_size=config['vocab_size'],
        prefix_len=config['prefix_len'],
        payload_len=config['payload_len'],
        payload_seed=payload_seed,
        shuffle_seed=shuffle_seed
    )

    tester = CapacityTester(
        model_configs=config,
        dataset=dataset,
        start_n=tester_config.get('base_start_n', 128),
        search_tolerance=tester_config.get('search_tolerance', 0.05),
        batch_size=tester_config.get('batch_size', 128),
        max_epochs=tester_config.get('max_epochs', 500),
        init_seed=init_seed,
        device=device
    )
    total_params = tester.total_params
    history_log = tester.run()

    del df, dataset, tester
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return total_params, history_log


if __name__ == '__main__':
    args_json = sys.argv[1]
    args = json.loads(args_json)

    model_class_str = args["config"]["model_class"]
    args["config"]["model_class"] = getattr(model, model_class_str)

    total_params, history = run_single_experiment(config=args["config"], tester_config=args["tester_config"],
                                                  payload_seed=args["payload_seed"], shuffle_seed=args["shuffle_seed"],
                                                  init_seed=args["init_seed"], device=args.get("device", "cpu"))

    with open(args["save_path"], "w") as f:
        json.dump({
            "architecture": args["config"]["model_class"].__name__,
            "config": {k: v for k, v in args["config"].items() if k != "model_class"},
            "parameters": total_params,
            "seeds": {
                "payload": args["payload_seed"],
                "shuffle": args["shuffle_seed"],
                "init": args["init_seed"]
            },
            "history": history
        }, f, indent=4)
    print(f"-> Run {args['save_path']} saved.")
