"""
Document 6 orchestrator — builds data.yaml, trains YOLOv8n on the fashion
dataset, evaluates on the validation split, prints a summary.

Run with:
    python -m src.yolo_training.main
    python -m src.yolo_training.main --fraction 0.02 --epochs 1   # smoke test
"""

import argparse

from . import config
from .dataset_yaml import build_data_yaml
from .evaluate import evaluate_model
from .train import train_model


def run(epochs: int = config.EPOCHS, batch: int = config.BATCH, fraction: float = 1.0) -> None:
    data_yaml_path = build_data_yaml()
    print(f"data.yaml: {data_yaml_path}")

    device = config.get_device()
    print(f"device: {device}")

    training_result = train_model(
        data_yaml_path, epochs=epochs, batch=batch, fraction=fraction, device=device,
    )
    print(f"\nTrained {training_result.epochs_run} epochs in {training_result.train_time_s:.1f}s")
    print(f"Best weights: {training_result.best_weights_path}")

    eval_result = evaluate_model(training_result.best_weights_path, data_yaml_path, device=device)
    print(f"\nValidation — mAP50: {eval_result.map50:.3f} | mAP50-95: {eval_result.map50_95:.3f} "
          f"| precision: {eval_result.precision:.3f} | recall: {eval_result.recall:.3f}")
    print("\nPer-class mAP50:")
    for name, value in sorted(eval_result.per_class_map50.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    run(epochs=args.epochs, batch=args.batch, fraction=args.fraction)
