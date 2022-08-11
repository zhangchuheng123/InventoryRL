CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp18_40.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp18_50.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp18_60.yaml &