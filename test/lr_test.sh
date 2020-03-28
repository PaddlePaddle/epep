#!/bin/bash

root_dir=$(dirname $(readlink -f $0))/..

sed -i "s#fluid_bin=.*#fluid_bin=python#g" conf/var_sys.conf
#sed -i "s|cuda_lib_path|#cuda_lib_path|g" conf/var_sys.conf

#train
sh run.sh -c conf/linear_regression/linear_regression.local.conf

if [ $? -ne 0 ];then
    echo "[FATAL] $(date) lr train failure." >&2
    exit 1
fi

sleep 2

#test
sh run.sh -c conf/linear_regression/linear_regression.local.conf -m predict
