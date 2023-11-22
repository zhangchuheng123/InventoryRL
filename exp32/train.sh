CUDA_VISIBLE_DEVICES=0 python classical.py --config config/default.yaml --method ss_policy &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/default.yaml &