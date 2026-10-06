#!/usr/bin/env python3
"""set up key hyperparameters/arguments for training a hugging face model"""
import transformers


def configure_training_args(output_dir, epochs,
                            per_device_train_batch_size,
                            per_device_eval_batch_size,
                            learning_rate, weight_decay,
                            metric_for_best_model, seed):
    """set up key hyperparameters/arguments"""

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        metric_for_best_model=metric_for_best_model,
        seed=seed,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    return training_args
