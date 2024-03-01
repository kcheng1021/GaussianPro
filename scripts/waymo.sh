python train.py -s $path/to/data$ -m $save_path$ \
                --eval --position_lr_init 0.000016 --scaling_lr 0.001 --percent_dense 0.0005 --port 1021 --dataset waymo 

python render.py -m $save_path$
python metrics.py -m $save_path$

python train.py -s $path/to/data$ -m $save_path$ \
                --eval --flatten_loss --position_lr_init 0.000016 --scaling_lr 0.001 --percent_dense 0.0005 --port 1021 --dataset waymo\
                --sky_seg --normal_loss --depth_loss --propagation_interval 30 --depth_error_min_threshold 0.8 --depth_error_max_threshold 1.0 \
                --propagated_iteration_begin 1000 --propagated_iteration_after 12000 --patch_size 20 --lambda_l1_normal 0.001 --lambda_cos_normal 0.001
 
python render.py -m $save_path$
python metrics.py -m $save_path$
