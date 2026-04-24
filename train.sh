#!/bin/bash
#SBATCH --job-name=toolbox_preprocess
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=runs/slurm_logs/job_%j_%t.out
#SBATCH --error=runs/slurm_logs/job_%j_%t.err
#SBATCH --gres=gpu:a5000:1

source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate rppg-toolbox

cd /zfsauton2/home/psicinsk/mgr/rPPG-Toolbox

# python main.py --config_file ./configs/infer_configs/PURE_UNSUPERVISED.yaml
# python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml
# python main.py --config_file ./configs/infer_configs/SCAMPS_UNSUPERVISED.yaml
# python main.py --config_file ./configs/field_infer_configs/FIELD_UNSUPERVISED.yaml


# python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_RHYTHMFORMER_BASIC.yaml
# python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_RHYTHMFORMER_BASIC.yaml

python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml

# srun --exclusive -N1 -n1 python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_RHYTHMFORMER_BASIC.yaml &
# srun --exclusive -N1 -n1 python main.py --config_file ./configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml &

# srun --exclusive -N1 -n1 python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_RHYTHMFORMER_BASIC.yaml &
# srun --exclusive -N1 -n1 python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml &

# wait