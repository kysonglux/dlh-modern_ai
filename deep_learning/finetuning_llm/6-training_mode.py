#!/usr/bin/env python3
"""trains, evaluates and saves the model"""
import transformers


def train_model(model, training_args, train_dataset,
                eval_dataset, test_dataset, tokenizer,
                data_collator, compute_metrics, model_save_name):
    """trains, evaluates and saves the model"""

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    train_results = trainer.train()
    test_results = trainer.evaluate(test_dataset)
    trainer.save_model(model_save_name)
    tokenizer.save_pretrained(model_save_name)

    return trainer, train_results, test_results
