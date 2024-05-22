#!/bin/bash

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



export CUDA_VISIBLE_DEVICES=${devices=4,5}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=False \
    --fold=1 \
    --run=1
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=False \
    --fold=1 \
    --run=2
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=False \
    --fold=1 \
    --run=3
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=False \
    --fold=1 \
    --run=4
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=False \
    --fold=1 \
    --run=5
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=True \
    --fold=1 \
    --run=1
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=True \
    --fold=1 \
    --run=2
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=True \
    --fold=1 \
    --run=3
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=True \
    --fold=1 \
    --run=4
python3 -m crossbeam.experiment.run_crossbeam \
    --config="./crossbeam/experiment/deepcoder/configs/eval/lambdabeam_eval.py" \
    --domain="deepcoder" \
    --synthetic_test_tasks=True \
    --fold=1 \
    --run=5


