python train.py -s $path/to/data$ -m $save_path$ \
                --eval --position_lr_init 0.000016 --scaling_lr 0.001 --percent_dense 0.0005 --port 1021

python render.py -m $save_path$
python metrics.py -m $save_path$

python train.py -s $path/to/data$ -m $save_path$ \
                --eval --flatten_loss --position_lr_init 0.000016 --scaling_lr 0.001 --percent_dense 0.0005 --port 1021 \
                --normal_loss --depth_loss --propagation_interval 50 --depth_error_min_threshold 0.8 --depth_error_max_threshold 1.0 \
                --propagated_iteration_begin 1000 --propagated_iteration_after 6000 --patch_size 20 --lambda_l1_normal 0.001 --lambda_cos_normal 0.001

python render.py -m $save_path$
python metrics.py -m $save_path$

# normal_loss -- whether using planar-constrained loss
# depth_loss -- whether using propagation
# propagation_interval -- the frequency for activating propagation
# depth_error_min_threshold -- the min threshold of relative depth error between rendered depth and propagated depth for initializing new gaussians
# depth_error_max_threshold -- the max threshold of relative depth error between rendered depth and propagated depth for initializing new gaussians
# patch size for patchmatching, make it bigger if your scenes are consisted of many large textureless planes, smaller otherwise
# lambda_xx_normal normal loss weight
