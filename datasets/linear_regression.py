#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


"""
 Specify the brief linear_regression.py
 Date: 2019/08/26 20:36:18
"""

import os
import sys
import numpy as np
import random
import paddle.fluid as fluid

from datasets.base_dataset import BaseDataset


class LinearRegression(BaseDataset):
    """
    LinearRegression dataset 
    """
    def __init__(self, flags):
        super(LinearRegression, self).__init__(flags)

    def parse_context(self, inputs):
        """
        set inputs_kv: please set key as the same as layer.data.name

        notice:
        (1)
        If user defined "inputs key" is different from layer.data.name,
        the frame will rewrite "inputs key" with layer.data.name
        (2)
        The param "inputs" will be passed to user defined nets class through
        the nets class interface function : net(self, FLAGS, inputs), 
        """
        inputs['x'] = fluid.layers.data(name="x", shape=[self._flags.input_size], dtype="float32")
        inputs['y'] = fluid.layers.data(name="y", shape=[1], dtype="float32")

        context = {"inputs": inputs}

        #set debug list, print info during training
        #debug_list = [key for key in inputs]
        return context

    def parse_oneline(self, line):
        """
        parse sample 
        """
        cols = line.strip("\t\n").split("\t")

        #input_size is size of vector X, 1 is label.

        label = [0]
        
        if len(cols) >= self._flags.input_size + 1:
            label = [float(cols[self._flags.input_size])]
       
        input_list = [float(x) for x in cols[:self._flags.input_size]]


        yield ("x", input_list),\
              ("y", label)
        
