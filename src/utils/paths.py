# GROUP19_COMP0197
import os

# get base path
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
adl_team_grey_dir = os.path.dirname(os.path.dirname(curr_file_dir))

# get relative paths
src_dir = os.path.join(adl_team_grey_dir, "src")
baseline_dir = os.path.join(src_dir, 'baseline')
datasets_dir = os.path.join(src_dir, 'datasets')
animals_10_dir = os.path.join(datasets_dir, 'Animals-10')
imagenet_dir = os.path.join(datasets_dir, 'imagenet-1k-resized')
oxford_3_dir = os.path.join(datasets_dir, 'Oxford-3')
fine_tuning_dir = os.path.join(src_dir, 'finetuning')
models_dir = os.path.join(src_dir, 'models')
outputs_dir = os.path.join(src_dir, 'outputs')
mvae_dir = os.path.join(src_dir, 'mvae')
utils_dir = os.path.join(src_dir, 'utils')
