CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp26_20_20sku.yaml &
CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp26_30_20sku.yaml &
CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp26_40_20sku.yaml &

# CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp26_20.yaml &
# CUDA_VISIBLE_DEVICES=1 python sac_discrete.py --config config/exp26_30.yaml &
# CUDA_VISIBLE_DEVICES=0 python sac_discrete.py --config config/exp26_40.yaml &