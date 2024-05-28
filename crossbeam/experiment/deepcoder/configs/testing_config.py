from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  # Data & Primitives
  config.domain = 'deepcoder'

  # Data Generation
  config.data_gen_timeout = 30
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
  config.abstraction_refinement = True
  config.data_save_dir = "./neurips/testing/data"
  config.num_datagen_proc = 1
  config.data_gen_seed = 0
  config.num_searches = 1
  config.shard_size = 1000
  config.dynamic_time_increase = 0

  config.seed = 0
  config.tout = 3600
  config.domain = 'deepcoder'
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.grad_accumulate = 4
  config.beam_size = 10
  config.num_proc = 2
  config.gpu_list = "0, 2"
  config.gpu = 0
  config.embed_dim = 128
  config.eval_every = 10
  config.port = '30005'
  config.use_ur = False
  config.do_test = False
  config.timeout = 0.5
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
  config.json_results_file = "./neurips/testing/"
  config.save_dir = "./neurips/testing"

  # Abstraction
  config.abstraction = True
  config.dynamic_tasks = True
  config.num_starting_ops = 28
  config.use_ur_in_valid = True
  config.top_k = 2
  config.used_invs = None
  config.initialization_method = "top"
  config.abstraction_pruning = True
  config.num_inventions_per_iter = 99
  config.invention_arity = 3
  config.max_invention = 10
  return config
