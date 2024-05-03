from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  # Data & Primitives
  config.domain = 'deepcoder'

  # Data Generation
  config.data_gen_timeout = 1000
  config.max_num_examples = 5
  config.max_num_inputs = 3 # DreamCoder Evaluation Tasks only have 1 input, DeepCoder can have multiple
  config.min_num_examples = 2
  config.min_num_inputs = 1
  config.max_search_weight = 20
  config.min_task_weight = 3
  config.max_task_weight = 15
  config.num_tasks_per_weight = 800 # Adapt 
  config.skip_probability = 0.0
  config.lambda_skip_probability = 0.0
  config.lambda_fraction = 0.8
  config.shuffle_ops = False
  config.abstraction_refinement = False
  config.data_save_dir = "neurips/lambdabeam/data2"
  config.num_datagen_proc = 30
  config.data_gen_seed = 2
  config.num_searches = 300
  config.shard_size = 1000
  config.dynamic_time_increase = 200 # Too less, need more

  config.seed = 2
  config.tout = 3600
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.grad_accumulate = 4
  config.beam_size = 10
  config.num_proc = 4
  config.gpu_list = '0, 1, 2, 3'
  config.gpu = 1
  config.embed_dim = 128
  config.eval_every = 10000
  config.port = '30008'
  config.use_ur = False
  config.do_test = False
  config.timeout = 200
  config.restarts_timeout = 10
  config.encode_weight = True
  config.train_steps = 10000000
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.lr = 5e-4
  config.load_model = 'model-latest.ckpt'
  config.steps_per_curr_stage = 10000
  config.schedule_type = 'uniform'
  config.json_results_file = "neurips/lambdabeam/results2/run_1.json"
  config.save_dir = "/ceph/jzenkner/Code/AbstractBeam/neurips/lambdabeam/models2"

  # Abstraction
  config.abstraction = False
  config.abstract_every = 10000
  config.num_starting_ops = 28
  config.dynamic_tasks = True
  config.use_ur_in_valid = True
  config.initialization_method = "top"
  config.abstraction_pruning = True
  config.top_k = 2
  config.num_inventions_per_iter = 99
  config.invention_arity = 3 
  config.used_invs = None
  config.max_invention = 999
  config.castrate_macros = False
  return config
