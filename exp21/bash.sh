CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/new_exp21_200.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/new_exp21_300.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/new_exp21_400.yaml &