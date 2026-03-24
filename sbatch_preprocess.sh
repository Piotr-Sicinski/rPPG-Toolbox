#!/bin/bash
#SBATCH --job-name=toolbox_preprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm_logs/job_%j_%t.out
#SBATCH --error=runs/slurm_logs/job_%j_%t.err
#SBATCH --gpus-per-task=1

source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate rppg-toolbox

cd /zfsauton2/home/psicinsk/mgr/rPPG-Toolbox

# python main.py --config_file ./configs/infer_configs/PURE_UNSUPERVISED.yaml
# python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml
python main.py --config_file ./configs/field_infer_configs/FIELD_UNSUPERVISED.yaml

