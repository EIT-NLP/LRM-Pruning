#  Structure Pruning Evaluation: Classification, Generation, and Reasoning
Model Pruning can changes a model's structure, which can break compatibility with standard evaluation frameworks. This repository provides a unified evaluation setup to fairly and consistently assess structure-pruned models across tasks like classification, generation, and reasoning.

## Supported Pruning Methods
This repository supports three types of structure pruning methods:

Static Depth: Shortened-PPL, Shortened-Taylor

Static Width: LLM-Pruner, SliceGPT

Dynamic Depth: SkipGPT, DLLM, MOD

It provides a unified evaluation framework for all these pruning strategies across multiple tasks. We are continuously integrating more structured pruning methods to expand the evaluation coverage.

## Evaluation for Classifcation
Our evaluation framework is based on the lm-evaluation-harness, adapted and extended to handle structure-pruned models. These modifications enable fair and consistent assessment across different pruning strategies and tasks.
