# AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning

This repository contains the source code associated with the paper submited to NeurIPS 2024:

In this research project, we aim to reduce the search space blowup in Program Synthesis. For this purpose, we train a neural model to learn a search
policy for bottom-up execution-guided program synthesis and extend it using DSL Enhancement.



## Setup

For dependencies, first install the packages in the requirements file, e.g.:
```
pip install -r requirements.txt
```
## File structure

The synthetic training data is saved to  `./neurips/abstractbeam/data` and  `./neurips/lambdabeam/data`.
Make sure to also create `./neurips/abstractbeam/models` and `./neurips/abstractbeam/results` directories. Same goes when you want to train the LambdaBeam benchmark.

## Train the model
Navigate to `crossbeam/experiment/deepcoder` directory, make any necessary edits
to `run_deepcoder.sh` including the data, model, and result folder and number of GPUs to use.
To train a model run below from the project's root:

```
./crossbeam/experiment/deepcoder/run_deepcoder.sh
```

The default hyperparameters should mirror the settings in the paper.

## Evaluating the trained models
Adapt `./crossbeam/experiment/deepcoder/run_deepcoder.sh` by changing the training configs to the evaluation configs.
Also make sure to create a models, data, and results directory in `./neurips/abstractbeam/eval/` or `./neurips/lambdabeam/eval/`.
From the root directory, run:

```
./crossbeam/experiment/deepcoder/run_deepcoder.sh
```

