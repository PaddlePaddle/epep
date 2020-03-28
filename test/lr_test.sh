#!/bin/bash

sed -i "s#fluid_bin=.*#fluid_bin=python#g" conf/var_sys.conf
#sed -i "s|cuda_lib_path|#cuda_lib_path|g" conf/var_sys.conf

#train
sh run.sh -c conf/linear_regression/linear_regression.local.conf

sleep 2

#test
sh run.sh -c conf/linear_regression/linear_regression.local.conf -m predict
