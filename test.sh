#!/bin/bash
#SBATCH --error=error.txt
#SBATCH --partition=dev_gpu_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00

module load

python -c "import torch, os;\
    print('Device count ', torch.cuda.device_count());\
    print('Is cuda available? ',torch.cuda.is_available());\
    print(vars(os.environ));\
    print(torch.zeros((1,1)).cuda());"
sleep 30

echo "done"

                                                                                                                                                                                                                                                                                                                                                                                                21,36         Bot

