#!/bin/bash
#SBATCH --job-name=toolbox_preprocess
#SBATCH --partition=cpu 		# (General, Debug, Preempt or Cpu)
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm_logs/job_%j_%t.out
#SBATCH --error=runs/slurm_logs/job_%j_%t.err


source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate rppg-toolbox

cd /zfsauton2/home/psicinsk/mgr/rPPG-Toolbox

python main.py --config_file ./configs/infer_configs/PURE_UNSUPERVISED.yaml
python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml