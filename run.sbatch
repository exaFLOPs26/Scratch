#!/bin/bash

#SBATCH -J ddpg                    # Job 이름
#SBATCH -p suma_rtx4090
#SBATCH -q base_qos
#SBATCH -o ./output.out                     
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                      # Job에 사용할 리소스 (GPU)

# display가 없는 server에서 mujoco를 돌릴 때 발생하는 에러를 방지하기 위해 설정
export MUJOCO_GL=egl

python3 script/main.py
$@
