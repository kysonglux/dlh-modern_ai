#!/usr/bin/env python3

configure_training_args = __import__('5-configure_training_arguments').configure_training_args

training_args = configure_training_args(
    output_dir="./hbtn_finetuning_emotion_classifier",
    epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    seed=0
)

print(training_args)