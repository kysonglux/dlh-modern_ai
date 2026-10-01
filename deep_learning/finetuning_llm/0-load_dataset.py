#!/usr/bin/env python3
"""load the emotion dataset using hugging face datasets library"""
import datasets


def load_emotion_dataset():
    """
    Load the emotion dataset from hugging face datasets.
    """

    # Load the dataset
    dataset = datasets.load_dataset("dair-ai/emotion")

    # Split the dataset into train, validation, and test sets
    train = dataset['train']
    validation = dataset['validation']
    test = dataset['test']

    return dataset, train, validation, test
