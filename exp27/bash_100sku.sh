CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp27_20_100sku.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp27_30_100sku.yaml &
CUDA_VISIBLE_DEVICES=2 python sac_discrete.py --config config/exp27_40_100sku.yaml &