# CUDA_VISIBLE_DEVICES=2 python train.py --mutual 1 --epochs 2500 --data cub --batch_size 80 --num_instances 5 --lr 1e-5 --online_training True --net bn_inception --loss hard_triplet --save_step 50 --save_dir ckps/cub/bn_inception-dim-512-lr1e-5-batchsize-80-corrloss2-test --resume /media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80-2/ckp_ep1500.pth.tar

# CUDA_VISIBLE_DEVICES=0 python train.py --mutual 1 --epochs 450 --data cub --batch_size 48 --num_instances 3 --lr 1e-5 --online_training True --net vit --loss binomial --save_step 20 --save_dir ckps/cub/vit-dim-512-lr1e-5-batchsize-80-test --resume /media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/ckps/Binomial/cub/ViT-DIM-384-lr3e-5-ratio-0.16-BatchSize-64-test-origin/ckp_ep255.pth.tar

# CUDA_VISIBLE_DEVICES=2 python train.py --mutual 1 --epochs 3400 --data cub --batch_size 48 --num_instances 3 --lr 1e-5 --online_training True --net resnet50 --loss binomial --save_step 100 --save_dir ckps/cub/resnet50-dim-512-lr1e-5-batchsize-48-test --resume /media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/ckps/Binomial/cub/ResNet50-DIM-512-lr1e-5-ratio-0.16-BatchSize-48-origin/ckp_ep2550.pth.tar


# CUDA_VISIBLE_DEVICES=0 python train.py --mutual 1 --epochs 8200 --data cub --batch_size 80 --num_instances 5 --lr 1e-5 --online_training True --feature_estimation True --net bn_inception --loss hard_triplet --save_step 50 --save_dir ckps/cub/bn_inception-dim-512-lr1e-5-batchsize-80-multi-test-s4 --resume /media/data3/gdliu_data/Deep-Incremental-Image-Retrieval-main/ckps/HardMining/cub/BN_Inception-DIM-512-lr1e-5-ratio-0.16-BatchSize-80-three-estima-skip-s3/ckp_ep7150.pth.tar 


CUDA_VISIBLE_DEVICES=3 python train2.py --epochs 450 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --online_training True --net bn_inception --loss hard_mining --save_step 10 --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-lch+corr-s1 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-origin/ckp_ep170.pth.tar

# new split
# origin train

# CUDA_VISIBLE_DEVICES=1 python train.py --epochs 150 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 5  --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-all

# s1
# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 450 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s1 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-origin/ckp_ep170.pth.tar

# s2
# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 600 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True  --feature_estimation True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s2 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s1/ckp_ep360.pth.tar

# s3

# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 680 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True  --feature_estimation True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s3 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s2/ckp_ep480.pth.tar

# skip
# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 680 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True  --feature_estimation True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-skip-s3 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s2/ckp_ep480.pth.tar

#s4

# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 770 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True  --feature_estimation True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s4 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s3/ckp_ep570.pth.tar

# CUDA_VISIBLE_DEVICES=1 python train.py --mutual 1 --epochs 770 --data deep2 --batch_size 80 --num_instances 5 --lr 1e-5 --net bn_inception --loss hard_mining --save_step 10 --online_training True  --feature_estimation True --save_dir ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-skip-s4 --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-skip-s3/ckp_ep580.pth.tar