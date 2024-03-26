from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  config.sweep = [{'config.schedule_type': 'halfhalf-2'},
                  {'config.schedule_type': 'uniform'},
                  {'config.schedule_type': 'all-3'},
                  {'config.schedule_type': 'all-4'},
                  {'config.schedule_type': 'all-5'},
                  {'config.schedule_type': 'all-6'}]

  config.seed = 0
  config.tout = 3600
  config.domain = 'deepcoder'
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.min_num_examples = 2
  config.max_num_examples = 5
  config.min_num_inputs = 1
  config.max_num_inputs = 1
  config.max_search_weight = 20
  config.beam_size = 10
  config.num_proc = 1
  config.gpu_list = '0'
  config.gpu = 0
  config.embed_dim = 128
  config.port = '30000'
  config.use_ur = True
  config.stochastic_beam = False
  config.do_test = True
  config.synthetic_test_tasks = False
  config.json_results_file = ''
  config.timeout = 1000
  config.restarts_timeout = 6
  config.encode_weight = True
  config.train_steps = 1000000
  config.temperature = 1.0
  config.encode_weight = True
  config.train_data_glob = ''
  config.test_data_glob = ''
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.load_model = '/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/outputs/baseline/model-best-valid.ckpt' # here
  config.dreamcoder = True
  config.abstraction = False
  config.data_name = ""
  # "/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/t-3600-maxne-5-maxni-1-skip-0.0-lambdaskip-0.0-lambdafrac-0.8-shuffleops-False"
  # "/work/ldierkes/repos/ma-lukas-dierkes/ec/LambdaBeam/t-120-maxne-5-maxni-1-skip-0.0-lambdaskip-0.0-lambdafrac-0.8-shuffleops-False"
  return config
