#!/bin/bash

path="$(dirname $0)"

export PATH=~/.local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages/torch/lib
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234
export BYTEPS_LOG_LEVEL=WARNING

function cleanup() {
  rm -rf lr.s
}

trap cleanup EXIT

pkill bpslaunch
pkill python3
sleep 2

echo "Launch scheduler"
export DMLC_ROLE=scheduler
bpslaunch &

echo "Launch server"
export DMLC_ROLE=server
bpslaunch &

export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_ROLE=worker
export BYTEPS_THREADPOOL_SIZE=4
export BYTEPS_FORCE_DISTRIBUTED=1

if [ "$TEST_TYPE" == "mxnet" ]; then
  echo "TEST MXNET ..."
  bpslaunch python3 $path/test_mxnet.py $@
elif [ "$TEST_TYPE" == "keras" ]; then
  echo "TEST KERAS ..."
  python $path/test_tensorflow_keras.py $@
elif [ "$TEST_TYPE" == "onebit" ] || [ "$TEST_TYPE" == "topk" ] || [ "$TEST_TYPE" == "randomk" ] || [ "$TEST_TYPE" == "dithering" ]; then
  export BYTEPS_MIN_COMPRESS_BYTES=0
  export BYTEPS_PARTITION_BYTES=2147483647
  echo "TEST $TEST_TYPE"
  bpslaunch python3 test_$TEST_TYPE.py
else
  echo "Error: unsupported $TEST_TYPE"
  exit 1
fi
