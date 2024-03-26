# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from functools import partial
from crossbeam.model.op_arg import LSTMArgSelector, AttnLstmArgSelector
from crossbeam.model.op_init import PoolingState, OpPoolingState
from crossbeam.model.encoder import LambdaSigIOEncoder
from crossbeam.model.encoder import LambdaSigValueEncoder
from crossbeam.model.encoder import ValueWeightEncoder, DummyWeightEncoder
from crossbeam.algorithm.variables import MAX_NUM_FREE_VARS, MAX_NUM_BOUND_VARS


def lstm_arg_selector(args, embed_dim, step_score_func, score_normed):
    if args.arg_selector == 'lstm':
        return LSTMArgSelector(hidden_size=embed_dim,
                               mlp_sizes=[256, 1],
                               step_score_func=step_score_func,
                               step_score_normalize=score_normed)
    elif args.arg_selector == 'attn_lstm':
        return AttnLstmArgSelector(hidden_size=embed_dim,
                                   mlp_sizes=[256, 1],
                                   step_score_func=step_score_func,
                                   step_score_normalize=score_normed)
    else:
        raise ValueError('unknown arg selector %s' % args.arg_selector)



class DeepCoderModel(nn.Module):
  def __init__(self, args, operations, ckpt_inventions=[]):
    super(DeepCoderModel, self).__init__()
    self.ops = tuple(operations)
    self.ckpt_inventions = ckpt_inventions
    self.op_idx_map = {repr(op): i for i, op in enumerate(tuple(list(self.ops)))}
    print('op_idx_map', self.op_idx_map)
    self.use_op_specific_lstm = args.use_op_specific_lstm
    self.embed_dim = args.embed_dim
    self.step_score_func = args.step_score_func
    self.score_normed = args.score_normed
    

    if args.io_encoder == 'lambda_signature':
      self.io = LambdaSigIOEncoder(args.max_num_inputs, hidden_size=args.embed_dim)
    else:
      raise ValueError('unknown io encoder %s' % args.io_encoder)

    if args.value_encoder == 'lambda_signature':
      val = LambdaSigValueEncoder(hidden_size=args.embed_dim)
    else:
      raise ValueError('unknown value encoder %s' % args.value_encoder)
    if args.encode_weight:
      self.encode_weight = ValueWeightEncoder(hidden_size=args.embed_dim)
    else:
      self.encode_weight = DummyWeightEncoder()
    self.val = val
    self.fn_arg_mod = partial(lstm_arg_selector, args, embed_dim=self.embed_dim,
                                  step_score_func=self.step_score_func,
                                  score_normed=self.score_normed)
    init_mod = OpPoolingState
    self.init = init_mod(ops=tuple(list(self.ops)), state_dim=args.embed_dim, pool_method='mean')
    if self.use_op_specific_lstm:
      self.op_specific_lstm = nn.ModuleList([
        self.fn_arg_mod() for _ in range(len(operations))])
    else:
      self.lstm = self.fn_arg_mod()

    self.special_var_embed = nn.Parameter(torch.zeros(MAX_NUM_FREE_VARS + MAX_NUM_BOUND_VARS, args.embed_dim))
    nn.init.xavier_uniform_(self.special_var_embed)


  # Abstraction
  def add_invention(self, invention, outer_op, func_for_args, device, initialization_method):
    self.ops = tuple(list(self.ops) + [invention.name])
    self.init.add_invention(invention, outer_op, func_for_args, device, initialization_method)
    if self.use_op_specific_lstm:
      if initialization_method == "top" and outer_op is not None:
        module = copy.deepcopy(self.op_specific_lstm[self.op_idx_map[outer_op]])
        self.op_specific_lstm.append(module.to(device)) # self.fn_arg_mod().to(device)) 
      elif initialization_method == "average" or initialization_method == "top_and_avg":
        modules = []
        for func in func_for_args:
          modules.append(copy.deepcopy(self.op_specific_lstm[self.op_idx_map[func]]).to(device))
        
        if initialization_method == "top_and_avg" and outer_op is not None:
          modules.append(copy.deepcopy(self.op_specific_lstm[self.op_idx_map[outer_op]]).to(device))
        
        if len(modules) == 0:
          self.op_specific_lstm.append(self.fn_arg_mod().to(device))
        elif len(modules) == 1:
          self.op_specific_lstm.append(modules[0])
        else:
          new_model = copy.deepcopy(modules[0]).to(device)
          new_model_state_dict = new_model.state_dict()
          for module in modules[1:]:
            current_module_state_dict = module.state_dict()
            for key in new_model_state_dict:
              new_model_state_dict[key] = (new_model_state_dict[key] + current_module_state_dict[key]) / 2

          new_model.load_state_dict(new_model_state_dict)
          self.op_specific_lstm.append(new_model.to(device))
      else:
        self.op_specific_lstm.append(self.fn_arg_mod().to(device))
      self.op_idx_map[invention.name.split("(")[0]] = len(self.ops) - 1
  

  

  def arg(self, op=None):
    if self.use_op_specific_lstm:
      return self.op_specific_lstm[self.op_idx_map[repr(op)]]
    else:
      return self.lstm
