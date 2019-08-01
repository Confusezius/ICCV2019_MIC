### CUB200
python MIC_ICCV19_Main.py --gpu 0 --seed 1 --savename result_run_cub200   --dataset cub200 --shared_num_classes 30  --cluster_update_freq 3 --adv_weights 2500 --task_p 1 0.8 --tau 52 --n_epochs 110 --random_cluster_pick_p 0.2

### CARS196
python MIC_ICCV19_Main.py --gpu 0 --seed 2 --savename result_run_cars196  --dataset cars196 --shared_num_classes 200  --cluster_update_freq 3  --adv_weights 500 --task_p 1 0.8 --tau 65 --n_epochs 110 --random_cluster_pick_p 0.2

### Online Products
python MIC_ICCV19_Main.py --gpu 0 --seed 1 --savename result_run_sop      --dataset online_products --shared_num_classes 50   --cluster_update_freq 3 --random_cluster_pick_p 0.2 --adv_weights 250  --task_p 1 0.5 --tau 20 35 --n_epochs 45

### In-Shop Clothes
python MIC_ICCV19_Main.py --gpu 0 --seed 1 --savename result_run_inshop   --dataset in-shop --shared_num_classes 150 --cluster_update_freq 3 --random_cluster_pick_p 0.2 --adv_weights 250 --task_p 1 0.8 --tau 45 --n_epochs 70

### PKU Vehice-ID
python MIC_ICCV19_Main.py --gpu 0 --seed 1 --savename result_run_vhid     --dataset vehicle_id --shared_num_classes 50   --cluster_update_freq 3 --random_cluster_pick_p 0.2 --adv_weights 250  --task_p 1 0.5 --tau 20 35 --n_epochs 45






python main.py --gpu 0 --bs 60 --seed 1 --source_path /media/karsten_dl/QS/Data/Dropbox/Projects/Datasets --savename result_run_cub200   --dataset cub200 --shared_num_classes 30  --cluster_update_freq 3 --adv_weights 2500 --task_p 1 0.8 --tau 52 --n_epochs 110 --random_cluster_pick_p 0.2
