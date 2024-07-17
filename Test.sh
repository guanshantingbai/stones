cur=8900
end=8910
step=10

while [ $cur -lt $end ]
do
   CUDA_VISIBLE_DEVICES=3 \
   python test.py --data car --net bn_inception \
   --resume /home/gdliu/online-dml/ckps/car/bn_inception-hardmining-lr1e-5-batchsize-80-lch+corr/ckp_ep${cur}.pth.tar --gallery_eq_query True --dim 512 --batch_size 80
   cur=`expr $cur + $step`
done