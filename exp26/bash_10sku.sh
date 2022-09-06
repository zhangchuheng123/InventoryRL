CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp26_20_10sku.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp26_30_10sku.yaml &
CUDA_VISIBLE_DEVICES=2 python sac_discrete.py --config config/exp26_40_10sku.yaml &