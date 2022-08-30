CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp25_20.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp25_30.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp25_40.yaml &
# CUDA_VISIBLE_DEVICES=1 python classical.py --config config/exp24_eval.yaml --method ss_policy &