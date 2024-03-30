# Configuration Parameters

This markdown file documents the parameters used in a Python script for configuration purposes.

## Data Generation

- **Domain**: 'deepcoder' 
  -  Indicates the domain for which the configurations are set, either "deepcoder" or "dreamcoder".
  
- **Data Generation Timeout**: 1000 
  -  Timeout duration for data generation for one search. It may need to be increased for larger Domain Specific Languages (DSLs).
  
- **Max Number of Examples per Task**: 5 
  -  Maximum number of input/output examples per task.
  
- **Max Number of Inputs per Task**: 3 
  -  Maximum number of inputs allowed per task.
  
- **Min Number of Examples per Task**: 2 
  -  Minimum number of input/output examples per task.
  
- **Min Number of Inputs per Task**: 1 
  -  Minimum number of inputs allowed per task.
  
- **Max Search Weight**: 20 
  -  Maximum program length until the search is stopped. (Note: It will not reach 20 due to data generation timeout.)
  
- **Min Task Weight**: 3 
  -  Minimum program length for a task.
  
- **Max Task Weight**: 15 
  -  Maximum program length until the search is stopped. (Note: It will not reach 20 due to data generation timeout.)
  
- **Number of Tasks per Weight**: 800 
  -  Number of tasks per weight that will be sampled per search.
  
- **Skip Probability**: 0.0 
  -  Probability to skip tasks that do not include lambdas. (Not used in this configuration.)
  
- **Lambda Skip Probability**: 0.0 
  -  Probability to skip tasks including lambdas. (Not used in this configuration.)
  
- **Lambda Fraction**: 0.8 
  -  Percentage of tasks that must include a lambda.
  
- **Shuffle Operations**: True 
  -  Indicates whether to shuffle operations before running the search.
  
- **Data Save Directory**: "" 
  -  Directory to save generated data.
  
- **Number of Data Generation Processes**: 30 
  -  Number of parallel processes for data generation.
  
- **Data Generation Seed**: 2 
  -  Seed for data generation.
  
- **Number of Searches**: 300 
  -  Number of searches to be performed.
  
- **Shard Size**: 1000 
  -  Maximum number of tasks stored per file.
  
- **Dynamic Time Increase**: 30.0 
  -  Time increment for newfound operations.

## Search & Training

- **Seed**: 2 
  -  Random seed for the unique randomizer.
  
- **Timeout (Obsolete)**: 3600 
  -  Obsolete parameter, to be removed in future versions.
  
- **IO Encoder**: 'lambda_signature' 
  -  Encoder for input/output pairs.
  
- **Model Type**: 'deepcoder' 
  -  Model type, used for both dreamcoder and deepcoder domains.
  
- **Value Encoder**: 'lambda_signature' 
  -  Encoder for values.
  
- **Gradient Accumulation Across Searches**: 4 
  -  Number of searches for gradient accumulation.
  
- **Beam Size**: 10 
  -  Beam size for beam search.
  
- **Number of GPUs**: 4 
  -  Number of GPUs to be used.
  
- **GPU List**: '0, 1, 2, 3' 
  -  List of GPUs to be used.
  
- **Master GPU**: 1 
  -  Master GPU.
  
- **Embedding Dimension**: 128 
  -  Dimensionality of embeddings.
  
- **Evaluation Frequency**: 10000 
  -  Number of training steps until evaluation and abstraction are run.
  
- **Port for Distributed Training**: '30008' 
  -  Port for distributed training. Update this if errors occur during training.
  
- **Unique Randomizer Usage**: False 
  -  If set to False, beam search will be used during training.
  
- **Test Evaluation**: False 
  -  Whether to perform test evaluation.
  
- **Max Evaluation Timeout per Task**: 100 
  -  Maximum evaluation timeout for one task.
  
- **Restarts Timeout**: 10 
  -  Time for each search, used for restarts.
  
- **Encode Intermediate Value Weight**: True 
  -  Whether to encode the weight of intermediate values.
  
- **Max Training Steps**: 10000000 
  -  Maximum number of training steps.
  
- **Use Op-Specific LSTM**: True 
  -  If True, every operation has its own Argument Selector Module.
  
- **Learning Rate**: 5e-4 
  -  Learning rate.
  
- **Load Model**: "" 
  -  Model to load. Leave empty.
  
- **Steps per Current Stage**: 10000 
  -  Parameter from LambdaBeam, not used.
  
- **Schedule Type for Task Ordering**: 'uniform' 
  -  Schedule type for task ordering.
  
- **JSON Results File**: "" 
  -  File to save results.
  
- **Save Directory for Model**: "" 
  -  Directory to save the trained model.

## Abstraction

- **Abstraction Usage**: True 
  -  Whether abstraction is used.
  
- **Number of Starting Operations**: 28 
  -  Number of starting operations for abstraction.
  
- **Dynamic Tasks Generation**: True 
  -  Whether dynamic tasks generation is used.
  
- **Unique Randomizer in Validation**: True 
  -  Whether unique randomizer is used in validation.
  
- **Initialization Method for Abstraction Models**: "top" 
  -  Initialization method for abstraction models.
  
- **Abstraction Pruning**: True 
  -  Whether abstraction pruning is enabled.
  
- **Top k Shortest Programs for Abstraction**: 2 
  -  Top k shortest programs to be used for abstraction phase.
  
- **Number of Abstractions Built per Iteration**: 999 
  -  Number of abstractions to be built per iteration.
  
- **Invention Arity**: 3 
  -  Arity of the invented operations.
  
- **Used Inventions**: None 
  -  If excluding inventions from the search is desired, specify here.
  
- **Max Inventions to Keep**: 999 
  -  Maximum number of abstractions to be kept.
