CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp23_20.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp23_30.yaml &
CUDA_VISIBLE_DEVICES=2 python sac_discrete.py --config config/exp23_40.yaml &