#!/bin/bash 
ip=`ifconfig ens3 | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`
port=1234
interface=eth0

repo_path='/home/ubuntu/repos/byteps'
script_path=$repo_path'/example/mxnet/train_gluon_imagenet_byteps_gc.py'
data_path='/home/ubuntu/data/ILSVRC2012/'

# args
algo=$1
lr=$2
model=resnet50_v2
epochs=120
batch_size=128
threadpool_size=16
omp_num_threads=4
min_compress_bytes=1024000

log_file=$algo"-"$lr
compression_args=''
if [[ $algo == "baseline" ]]; then
  $threadpool_size=0
elif [[ $algo == "onebit" ]]; then
  $compression_args='--compressor onebit --onebit-scaling --ef vanilla --compress-momentum nesterov'
elif [[ $algo == "topk" ]]; then
  k=$3
  $compression_args='--compressor topk --k '${k}' --ef vanilla --compress-momentum nesterov'
  $log_file=$log_file"-k="${k}
elif [[ $algo == "randomk" ]]; then
  k=$3
  $compression_args='--compressor randomk --k '${k}' --ef vanilla --compress-momentum nesterov'
  $log_file=$log_file"-k="${k}
elif [[ $algo == "dithering" ]]; then
  k=$3
  $compression_args='--compressor dithering --k '${k}' --normalize l2'
  $log_file=$log_file"-k="${k}
else
  echo "unknown compressor. aborted."
  exit
fi


cmd="python $repo_path/launcher/dist_launcher.py -WH hosts -SH hosts --scheduler-ip $ip --scheduler-port $port --interface $interface --username ubuntu --env OMP_WAIT_POLICY:PASSIVE OMP_NUM_THREADS:$omp_num_threads BYTEPS_THREADPOOL_SIZE:$threadpool_size BYTEPS_MIN_COMPRESS_BYTES:$min_compress_bytes BYTEPS_NUMA_ON:1 \"bpslaunch python3 $script_path --model $model --mode hybrid --rec-train $data_path"train.rec" --rec-train-idx $data_path"train.idx" --rec-val $data_path"val.rec" --rec-val-idx $data_path"val.idx" --use-rec --batch-size $batch_size --num-gpus 1 --num-epochs $epochs -j 2 --warmup-epochs 5 --warmup-lr $lr --lr $lr --lr-mode cosine --logging-file $log_file\""

echo $cmd
exec $cmd