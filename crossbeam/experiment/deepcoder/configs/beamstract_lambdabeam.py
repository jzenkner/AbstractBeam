from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  # Data & Primitives
  config.dreamcoder = False

  # Data Generation
  config.data_gen_timeout = 1000
  config.max_num_examples = 5
  config.max_num_inputs = 3
  config.min_num_examples = 2
  config.min_num_inputs = 1
  config.max_search_weight = 20
  config.min_task_weight = 3
  config.max_task_weight = 15
  config.num_tasks_per_weight = 800
  config.skip_probability = 0.0
  config.lambda_skip_probability = 0.0
  config.lambda_fraction = 0.8
  config.shuffle_ops = False
  config.data_save_dir = "/work/ldierkes/repos/new/LambdaBeam/outputs/beamstract/lambdabeam/top/data"
  config.num_datagen_proc = 35
  config.data_gen_seed = 0
  config.num_searches = 350
  config.shard_size = 1000

  config.seed = 0
  config.tout = 3600
  config.domain = 'deepcoder_lambdabeam'
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.grad_accumulate = 4
  config.beam_size = 10
  config.num_proc = 1
  config.gpu_list = "1"
  config.gpu = 1
  config.embed_dim = 64
  config.eval_every = 10000
  config.num_valid = 76
  config.port = '30000'
  config.use_ur = False
  config.do_test = False
  config.timeout = 100
  config.restarts_timeout = 10
  config.encode_weight = True
  config.train_steps = 10000000
  config.train_data_glob = None
  config.test_data_glob = None
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.lr = 5e-3
  config.load_model = ''
  config.steps_per_curr_stage = 5000
  config.schedule_type = 'uniform'
  config.json_results_file = "/work/ldierkes/repos/new/LambdaBeam/outputs/beamstract/lambdabeam/top/results.json"
  config.save_dir = "/work/ldierkes/repos/new/LambdaBeam/outputs/beamstract/lambdabeam/top"
  config.data_name = "/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/t-3600-maxne-5-maxni-1-skip-0.0-lambdaskip-0.0-lambdafrac-0.8-shuffleops-False-now/filtered_training"

  # Abstraction
  config.abstraction = True
  config.top_k = 2
  config.num_starting_ops = 28
  config.dynamic_tasks = True
  config.use_ur_in_valid = True
  config.initialization_method = "top"
  config.abstraction_pruning = True
  config.num_inventions_per_iter = 15
  config.invention_arity = 3
  config.max_invention = 10
  return config
