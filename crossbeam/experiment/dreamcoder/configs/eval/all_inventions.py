from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  # Data & Primitives
  config.domain = 'dreamcoder'

  # Dynamic Data Generation
  config.data_gen_timeout = 1000
  config.max_num_examples = 15
  config.max_num_inputs = 1
  config.min_num_examples = 5
  config.min_num_inputs = 1
  config.max_search_weight = 25
  config.min_task_weight = 3
  config.max_task_weight = 15
  config.num_tasks_per_weight = 800
  config.skip_probability = 0.0
  config.lambda_skip_probability = 0.0
  config.lambda_fraction = 0.8
  config.shuffle_ops = True
  config.data_save_dir = "/work/ldierkes/repos/new/LambdaBeam/outputs/all_inventions/data"
  config.num_datagen_proc = 35
  config.data_gen_seed = 0
  config.num_searches = 350
  config.shard_size = 1000

  # Search
  config.seed = 22
  config.tout = 3600
  
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.grad_accumulate = 4
  config.beam_size = 10
  config.num_proc = 1
  config.gpu_list = '6'
  config.gpu = 6
  config.embed_dim = 128
  config.eval_every = 10000
  config.num_valid = 76
  config.port = '30000'
  config.use_ur = True
  config.do_test = True
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
  config.json_results_file = "/work/ldierkes/repos/new/LambdaBeam/outputs/all_inventions/eval/results.json"
  config.save_dir = "/work/ldierkes/repos/new/LambdaBeam/outputs/all_inventions"

  # Abstraction
  config.abstraction = False
  config.dynamic_tasks = True
  config.num_starting_ops = 17
  config.use_ur_in_valid = True
  config.initialization_method = "top"
  config.abstraction_pruning = True
  config.num_inventions_per_iter = 99
  config.invention_arity = 3
  config.used_invs = None
  return config
