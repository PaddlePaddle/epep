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
 Date: 2019/08/26 20:56:18
"""
import math
import numpy as np
import logging
import collections

import paddle.fluid as fluid

from nets.base_net import BaseNet


class LinearRegression(BaseNet):
    """
    This module provide nets for linear_regression
    """
    def __init__(self, FLAGS):
        super(LinearRegression, self).__init__(FLAGS)

    def net(self, inputs):
        """
        linear regression interface
        """
        y_predict = fluid.layers.fc(input=inputs["x"], size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=inputs["y"]) 
        avg_cost = fluid.layers.mean(cost)
        
        # debug output info during training
        debug_output = collections.OrderedDict()
        model_output = {}
        net_output = {"debug_output": debug_output, 
                      "model_output": model_output}
        
        if self.is_training:
            net_output["loss"] = avg_cost
        
        debug_output['y'] = inputs["y"]
        debug_output['y_predict'] = y_predict

        model_output['feeded_var_names'] = ["x"]
        model_output['fetch_targets'] = [y_predict]

        return net_output

    def train_format(self, result, global_step, epoch_id, batch_id):
        """
            result: one batch train narray
        """ 
        if global_step == 0 or global_step % self._flags.log_every_n_steps != 0:
            return
        
        #result[0] default is loss.
        avg_res = np.mean(np.array(result[0]))
        vec = []
        for i in range(1, len(result)):
            res = np.array(result[i])
            vec.append("%s#%s" % (res.shape, ' '.join(str(j) for j in res.flatten())))
        logging.info("epoch[%s], global_step[%s], batch_id[%s], extra_info: "
                "loss[%s], debug[%s]" % (epoch_id, global_step, batch_id,
                avg_res, ";".join(vec)))

    def init_params(self, place):
        """
            init embed
        """
        def _load_parameter(pretraining_file, vocab_size, word_emb_dim):
            pretrain_word2vec = np.zeros([vocab_size, word_emb_dim], dtype=np.float32)
            for line in open(pretraining_file, 'r'):
                id, _, vec = line.strip('\r\n').split('\t')
                pretrain_word2vec[int(id)] = map(float, vec.split())

                return pretrain_word2vec

        embedding_param = fluid.global_scope().find_var("wordid_embedding").get_tensor()
        pretrain_word2vec = _load_parameter(self._flags.init_train_params,
                self._flags.vocab_size, self._flags.emb_dim)
        embedding_param.set(pretrain_word2vec, place)
        logging.info("init pretrain word2vec:%s" % self._flags.init_train_params)

    def pred_format(self, result, **kwargs):
        """
            format pred output
        """
        if result is None:
            return
    
        if result == '_PRE_':
            logging.info("init context before predict.") 
            return

        if result == '_POST_':
            if self._flags.init_pretrain_model is not None:
                path = "%s/infer_model" % (self._flags.export_dir)
                frame_env = kwargs['frame_env']
                fluid.io.save_inference_model(path,
                       frame_env.paddle_env['feeded_var_names'],
                       frame_env.paddle_env['fetch_targets'],
                       frame_env.paddle_env['exe'], frame_env.paddle_env['program'])

            return

        def _flat_str(vec):
            if isinstance(vec, (list, tuple, np.ndarray)): 
                return "%s:%s" % (np.array(vec).shape,
                            ";".join([' '.join(str(j) for j in np.array(i).flatten()) for i in vec]))
            else:
                return str(vec)

        out = '\t'.join([_flat_str(np.array(o)) for o in result])
        print(out) 

