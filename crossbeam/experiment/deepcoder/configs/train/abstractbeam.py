from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  # Data & Primitives
  config.domain = 'deepcoder' # Domain, set to "deepcoder" or "dreamcoder"

  # Data Generation
  config.data_gen_timeout = 1000 # Data generation timeout for one search - might need to be increased as bigger DSL
  config.max_num_examples = 5 # Max number of examples per task
  config.max_num_inputs = 3 # Max number of input per task 
  config.min_num_examples = 2 # Min number of examples per task
  config.min_num_inputs = 1 # Min number of inputs per task
  config.max_search_weight = 10 # Max program length until search is stopped (20 will not be reached due to data_gen_timeout)
  config.min_task_weight = 3 # Minimum program length
  config.max_task_weight = 10 # Max program length until search is stopped (20 will not be reached due to data_gen_timeout)
  config.num_tasks_per_weight = 800 # Number of tasks per weight that will be sampled per search
  config.skip_probability = 0.0 # Skip probability for tasks not including lambdas (never used that)
  config.lambda_skip_probability = 0.0  #Skip probability for tasks including lambdas (never used that)
  config.lambda_fraction = 0.8 # Percentage of tasks that need to include a lambda
  config.shuffle_ops = True # Shuffle operations before running search
  config.abstraction_refinement = True
  config.data_save_dir = "neurips/abstractbeam/data5" # Directory to save data
  config.num_datagen_proc = 30 # Number of parallel processes
  config.data_gen_seed = 2 # Data Generation Seed (will be incremented every iteration automatically)
  config.num_searches = 300 # Number of searches that will be performed
  config.shard_size = 1000 # Max number of tasks that are stored per file 
  config.dynamic_time_increase = 800 # Time increasment for newfound operations

  config.seed = 2 # Random seed for Unique Randomizer
  config.tout = 3600 # Obsolete (will be removed in future versions)
  config.io_encoder = 'lambda_signature' # Encoder for input/output pairs
  config.model_type = 'deepcoder' # Model type (use deepcoder for both dreamcoder and deepcoder domain)
  config.value_encoder = 'lambda_signature' # Encoder for values
  config.grad_accumulate = 4 # Gradient accumulation across searches. 
  config.beam_size = 10  # Beam size for beam search 
  config.num_proc = 4 # Number of GPUs
  config.gpu_list = '0,1,2,3' # GPU List
  config.gpu = 1 # Master GPU
  config.embed_dim = 128 # Embedding dimension
  config.eval_every = 10000 # Number of train steps until eval + abstraction will be run
  config.port = '30008' # Port for distributed training (sometimes it helped to updated this one when error occurs during training)
  config.use_ur = False # Unique Randomizer, if set to False, beam search will be used during training
  config.do_test = False # Test evaluation
  config.timeout = 100 # Max evaluation timeout for one task
  config.restarts_timeout = 10 # Time for each search (100 / 10 --> 10 restarts)
  config.encode_weight = True # Encode the weight of a intermediate value (e.g. Add(1, 2) has weight 3) --> show how "costly" a produced value is
  config.train_steps = 10000000 # Maximal train steps (will not be reached :))
  config.random_beam = False # I think is not implemented --> cannot be used
  config.use_op_specific_lstm = True # If True, every operation has its own Argument Selector Module (never tried setting to False)
  config.lr = 5e-4 # Learning rate
  config.load_model = 'model-latest.ckpt'  # Leave empty
  config.steps_per_curr_stage = 10000 # Parameter from LambdaBeam, we dont use this
  config.schedule_type = 'uniform' # Schedule type for task ordering
  config.json_results_file = "neurips/abstractbeam/results5/run_1.json" # File to save results
  config.save_dir = "/ceph/jzenkner/Code/AbstractBeam/neurips/abstractbeam/models5" # Directory to save model

  # Abstraction
  config.abstraction = True # Abstraction usage
  config.num_starting_ops = 28 # Number of starting operations
  config.dynamic_tasks = True # Dynamic tasks Generation 
  config.use_ur_in_valid = True # Unique Randomizer in validation
  config.initialization_method = "top" # Initialization method for abstraction models
  config.abstraction_pruning = True # Abstraction pruning, throw away useless abstractions like Add(x1, x2)
  config.top_k = 2 # Top k shortest program that will be used for abstraction phase 
  config.num_inventions_per_iter = 999 # How many abstraction should be build
  config.invention_arity = 2 # Arity of the invented operations
  config.used_invs = None # If u want to exclude inventions from the search, e.g. during testing only use abstractions that found during training
  config.max_invention = 999 # How many abstractions should be kept (top once are kept)
  config.castrate_macros = False
  return config
