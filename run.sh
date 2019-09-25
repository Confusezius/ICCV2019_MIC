#!/usr/bin/env bash
#NOTE: Most likely, the --source_path - flag will need to be set to the folder containing the datasets, e.g.
#--source_path /home/<user>/Datasets
SOURCE="/workplace/data/motion_efs/home/biagib/data/"
DEST="/workplace/data/motion_efs/home/biagib/MIC/"

### CUB200
python3 main.py --seed 0 --savename result_run_cub200_longer --dataset cub200 \
               --shared_num_classes 30  --cluster_update_freq 3 --adv_weights 2500 \
               --task_p 1 0.8 --tau 50  --n_epochs 200 --random_cluster_pick_p 0.2 \
               --source_path ${SOURCE} --save_path ${DEST}

### CARS196
#python3 main.py --seed 0 --savename result_run_cars196 --dataset cars196 \
#               --shared_num_classes 150  --cluster_update_freq 3 --adv_weights 500  \
#               --task_p 1 0.8 --tau 60   --n_epochs 110 --random_cluster_pick_p 0.2\
#               --source_path ${SOURCE} --save_path ${DEST}


#### Online Products
#python3 main.py --gpu 6 --seed 0 --savename result_run_sop --dataset online_products \
#               --shared_num_classes 50   --cluster_update_freq 3 --adv_weights 250  \
#               --task_p 1 0.5 --tau 20 35 --n_epochs 45 --random_cluster_pick_p 0.2
#
