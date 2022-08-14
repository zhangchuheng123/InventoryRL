CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp19_40.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp19_50.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp19_30.yaml &