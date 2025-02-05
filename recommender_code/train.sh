# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=Beauty --train_dir=default --maxlen=20 --device=cuda --alpha=0.1 --beta=0.1 --k_num=10