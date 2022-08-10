CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp17_5.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp17_10.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp17_20.yaml &