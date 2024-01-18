seeds=(10 88 400 1024 2023)

for ((i=0; i<5; i++))
do

python train.py  \
    --config_file configs/models/rn50_ep50.yaml \
    --dataset_config_file configs/datasets/coco.yaml \
    --input_size 224  \
    --lr 0.02  \
    --mlplr 0.0005 \
    --output_dir /data/your_model_name/coco_r${i} \
    --max_epochs 10 \
    --device_id 2 \
    --n_ctx 16 \
    --pool_size 8 \
    --beta 1 \
    --gamma 1 \
    --cln 1 \
    --prompt_key_init uniform \
    --learnable_alpha 1 \
    --seed ${seeds[i]} \
    --labels_file /labels.json \
    --uncertainty_weight 1 \
    --use_pfa 1 \
    --pfa_lr 0.0001 \
    --pfa_3d_key_smoothing 1 \
    --use_img_proto 1 \
    --use_text_proto 1 \

done