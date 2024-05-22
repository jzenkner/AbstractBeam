# AbstractBeam: Enhancing Bottom-Up Program Synthesis using Library Learning

This repository contains the source code associated with the paper submited to NeurIPS 2024:

In this research project, we aim to reduce the search space blowup in Program Synthesis. For this purpose, we train a neural model to learn a search
policy for bottom-up execution-guided program synthesis and extend it using DSL Enhancement.



## Setup
Make sure to install pytorch and pytorch-scatter.
We used torch==2.2.1 and torch-scatter==2.1.2.
Then install the packages in the requirements file, e.g.:
```
pip install -r requirements.txt
```
## File structure

The synthetic training data is saved to  `./neurips/abstractbeam/data` and  `./neurips/lambdabeam/data`.
Make sure to also create `./neurips/abstractbeam/models` and `./neurips/abstractbeam/results` directories. Same goes when you want to train the LambdaBeam benchmark.

## Train or eval the model
Navigate to `crossbeam/experiment/deepcoder` directory, and select the config you want to run.
Just adapt the config path to point to `./crossbeam/experiment/deepcoder/configs/` + [`train/abstractbeam.py`, `train/baseline.py`, `eval/abstractbeam_eval.py`, `eval/lambdabeam_eval.py`].
You can make any necessary edits to the selected config file including the data, model, and result directories.
Moreover, you can adapt the hyperparameters, e.g., the enumeration timeout, number of GPUs to use, ... .
To start run below from the project's root (the number of GPUs set in the config file must align with the number selected in below script):

```
./crossbeam/experiment/deepcoder/run_deepcoder.sh
```

