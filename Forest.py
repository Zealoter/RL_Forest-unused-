#!/usr/bin/env Python
# coding=utf-8
"""
@Copyright ©2019 JuQi
@Author: JuQi
@Contact: 964950472@qq.com
@Software: PyCharm
@File: Forest.py
@Time: 2020/9/30 6:22 下午
@Introduction:
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
import copy
from collections import deque
import random
import control
import time
import joblib
from joblib.parallel import Parallel, delayed
import os
from control import show_men_use
import gc


class Tree(object):
    def __init__(self, data, target, max_depth, action_len, use_feature_num):
        self.max_depth = max_depth
        self.action_len = action_len
        self.use_feature_num = use_feature_num
        self.reality_tree = DecisionTreeRegressor(
            max_depth=self.max_depth
        )
        self.use_feature = np.arange(self.action_len - 2)
        np.random.shuffle(self.use_feature)
        self.use_feature = self.use_feature[:self.use_feature_num]
        self.use_feature[self.use_feature_num - 2] = self.action_len - 2
        self.use_feature[self.use_feature_num - 1] = self.action_len - 1
        data = np.array(data)
        target = np.array(target)

        self.reality_tree.fit(data[:, self.use_feature], target)
        self.visit = 0
        self.evaluate_value = 0
        self.difference_value = 0

    def predict(self, data):
        data = np.array(data)
        data.reshape(-1, control.action_len)
        ans = self.reality_tree.predict(data[:, self.use_feature])
        return ans

    def train(self, data):
        data = np.array(data)
        return self.reality_tree.predict(data[:, self.use_feature])

    ''''''


def get_sample(tmp_data):
    tmp_batch = random.sample(tmp_data, control.train_sample_data_len)
    return tmp_batch


def parallel_get_tree_predict(tmp_trees, data):
    tmp_sum = []
    for tmp_i_tree in tmp_trees:
        tmp_sum.append(tmp_i_tree.train(data))
    return tmp_sum


class Forest(object):
    def __init__(
            self,
            n_estimators=100,
            action_len=control.action_len,
            max_depth=15,
            use_feature_num=25,
    ):
        self.trees = []
        self.n_estimators = np.int(n_estimators)
        self.action_len = action_len
        self.max_depth = max_depth
        self.use_feature_num = use_feature_num

    def init_random_forest(self, data, random_col=5000, sample_len=100, class_kind=5):
        tmp_forest = []
        if data:
            for _ in range(self.n_estimators):
                init_data_target = get_sample(data)
                use_data, use_target = zip(*init_data_target)
                tmp_forest.append(Tree(use_data, use_target))
        else:
            init_data = np.random.randint(2, size=(random_col, self.action_len))
            init_target = (np.random.rand(random_col) - 0.5) * class_kind

            for _ in range(self.n_estimators):
                choose_col = np.random.choice(a=init_data.shape[0], size=sample_len)
                use_data = init_data[choose_col, :]
                use_target = init_target[choose_col]
                tmp_forest.append(
                    Tree(
                        use_data,
                        use_target,
                        self.max_depth,
                        self.action_len,
                        self.use_feature_num
                    )
                )
        self.trees = tmp_forest

    def forest_predict(self, data, is_mean=True):
        tmp_sum = []
        for i_tree in range(self.n_estimators):
            tmp_sum.append(self.trees[i_tree].train(data))
        tmp_sum = np.array(tmp_sum)
        if is_mean:
            return np.mean(tmp_sum, axis=0)
        else:
            return tmp_sum

    def get_idx(self, evaluate_base):
        idx_evaluate_base = np.arange(self.n_estimators)
        return idx_evaluate_base[evaluate_base.argsort()]

    def train(self, data_per_train, kill):
        # show_men_use()
        start = time.time()

        data, target = zip(*data_per_train)
        max_play = -1

        tmp_tmp_sum = Parallel(n_jobs=control.train_use_cup_num, backend="multiprocessing")(
            delayed(parallel_get_tree_predict)(
                copy.deepcopy(
                    self.trees[i_tree * control.train_tree_part_len:(i_tree + 1) * control.train_tree_part_len]
                ),
                copy.deepcopy(data)
            ) for i_tree in range(control.train_use_cup_num)
        )
        tmp_sum = []
        for i_tmp_tmp_sum in tmp_tmp_sum:
            tmp_sum.extend(i_tmp_tmp_sum)

        end = time.time()
        print('1时间', end - start)

        for i_tree in range(self.n_estimators):
            if self.trees[i_tree].visit > max_play:
                max_play = self.trees[i_tree].visit
        max_play += 1

        tmp_sum = np.array(tmp_sum)
        target = np.array(target)

        # 现有的预测和训练集的差异
        mean_ans = np.mean(tmp_sum, axis=0)
        total_ans = mean_ans * self.n_estimators
        total_bias = np.mean((mean_ans - target) ** 2)

        evaluate_base = np.zeros(self.n_estimators)  # 去掉這一颗树的分數(越大越好)
        bias_base = np.zeros(self.n_estimators)  # 树和目標的距離(越小越好)
        difference_base = np.zeros(self.n_estimators)  # 树和其余树的距離(越大越好)

        evaluate_history = np.zeros(self.n_estimators)
        e_total_evaluate = np.zeros(self.n_estimators)
        d_total_evaluate = np.zeros(self.n_estimators)

        # 用相对排名(不用绝对数据)代表分数
        for i_tree in range(self.n_estimators):
            reject_ans = (total_ans - tmp_sum[i_tree, :]) / (self.n_estimators - 1)
            evaluate_base[i_tree] = np.mean(np.abs(reject_ans - target))
            bias_base[i_tree] = np.mean(np.abs(tmp_sum[i_tree, :] - target))
            difference_base[i_tree] = np.mean(np.abs(tmp_sum[i_tree, :] - mean_ans))

        mean_bias = np.mean(bias_base)  # 用作分数权重
        mean_difference = np.mean(difference_base)  # 用作分数权重

        good_idx_evaluate_base = self.get_idx(evaluate_base)  # 小的在前(好的在後)
        good_idx_bias_base = self.get_idx(bias_base)  # 小的在前(好的在前)
        good_idx_difference_base = self.get_idx(difference_base)  # 小的在前(好的在後)

        for i_tree_idx in range(self.n_estimators):
            op_point = i_tree_idx / self.n_estimators  # 随着排名增加，分数增加
            ne_point = 1 - i_tree_idx / self.n_estimators  # 随着排名增加，分数减少
            e_total_evaluate[good_idx_evaluate_base[i_tree_idx]] += (
                    op_point * (mean_difference + mean_bias))
            d_total_evaluate[good_idx_bias_base[i_tree_idx]] += (ne_point * mean_bias)
            d_total_evaluate[good_idx_difference_base[i_tree_idx]] += (op_point * mean_difference)

        for i_tree in range(self.n_estimators):
            self.trees[i_tree].evaluate_value = (self.trees[i_tree].evaluate_value * self.trees[
                i_tree].visit + e_total_evaluate[i_tree]) / (self.trees[i_tree].visit + 1)

            self.trees[i_tree].difference_value = (self.trees[i_tree].difference_value * self.trees[
                i_tree].visit + d_total_evaluate[i_tree]) / (self.trees[i_tree].visit + 1)

            self.trees[i_tree].visit += 1
            evaluate_history[i_tree] = self.trees[i_tree].difference_value + self.trees[
                i_tree].evaluate_value + control.C * np.sqrt(np.log(max_play) / self.trees[i_tree].visit)

        print('平均值分数:', np.mean(evaluate_history))
        print('target平均距離:', np.mean(bias_base))
        print('difference_base平均距離:', np.mean(difference_base))
        print('target平均距離和difference的差距:', np.mean(bias_base) - np.mean(difference_base))

        # 删除差的树
        idx = np.arange(self.n_estimators)
        good_idx = idx[evaluate_history.argsort()][kill:]
        new_trees = [self.trees[i_tree] for i_tree in good_idx]
        self.trees = new_trees

        # 计算剩下的树和目标的差距
        tmp_sum = tmp_sum[good_idx, :]
        need_fix_bias = (target - np.mean(tmp_sum, axis=0)) * (self.n_estimators - kill)
        # 计算所需修正
        data = np.reshape(data, (-1, self.action_len))
        fix_bias = need_fix_bias / (kill)
        target_change = target + fix_bias
        # 补全剩下的树
        for i_tree in range(kill):
            self.trees.append(
                Tree(
                    data,
                    target_change,
                    self.max_depth,
                    self.action_len,
                    self.use_feature_num
                )
            )

        return total_bias

    def get_data(self):
        """
        :return:
        """
        random_forest = [Forest(n_estimators=control.play_sample_tree_num) for _ in range(control.play_use_cup_num)]

        for i_forest in range(control.play_use_cup_num):
            random_forest[i_forest].trees = random.sample(self.trees, control.play_sample_tree_num)

        tmp_game_data = Parallel(n_jobs=control.play_use_cup_num, backend="multiprocessing")(
            delayed(sample_play_game)(random_forest[i_trees]) for i_trees in range(control.play_use_cup_num)
        )
        tmp_data = deque(maxlen=control.total_data_len)
        total_score = 0

        for i_game_data in tmp_game_data:
            tmp_data.extend(i_game_data[0])
            total_score += i_game_data[1]
        total_score /= control.play_use_cup_num
        print('这次平均分数:', total_score)
        return tmp_data

    def save_model(self, loop, dicti):
        joblib.dump(self.trees, dicti + '(' + str(loop) + ').pkl')

    def read_model(self, loop, dicti, is_refresh):
        tmp_dic = dicti + '(' + str(loop) + ').pkl'
        print(tmp_dic)
        self.trees = joblib.load(tmp_dic)
        if is_refresh:
            for i_tree in range(self.n_estimators):
                self.trees[i_tree].visit = 0
                self.trees[i_tree].evaluate_value = 0
                self.trees[i_tree].difference_value = 0


def sample_play_game(forest):
    # show_men_use('开始一个训练')
    game = Game(forest)
    score = 0
    for _ in range(control.one_cpu_once_play):
        score += game.run_game(
            {
                "players": 2,
                "random_start_player": True,
                "colors": 5,
                "rank": 5,
                "hand_size": 5,
                "max_information_tokens": 3
            }
        )
    score /= control.one_cpu_once_play
    return [game.game_data, score]
