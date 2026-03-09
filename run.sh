#!/bin/bash
#SBATCH --job-name=metric_eval
#SBATCH --account=rrg-annielee
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=metrics_%j.out
#SBATCH --error=metrics_%j.err

module load python/3.11 cuda/12.2 cudnn

if [ ! -d "venv" ]; then
    virtualenv --no-download venv
fi
source venv/bin/activate

pip install --no-index --upgrade pip
pip install pandas torch torchvision --no-index
pip install evaluate transformers datasets bert-score

pip install bleurt tensorflow
pip install unbabel-comet

python your_script.py

deactivate