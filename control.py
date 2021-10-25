#!/usr/bin/env Python
# coding=utf-8
import numpy as np
import psutil

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

action_len = 655
C = 0
alpha = 0.8
variance_decay = 1

train_loop = 1000
total_num = 400

play_use_cup_num = int(3)
play_sample_tree_num = 300
one_cpu_once_play = int(1)

train_use_cup_num = 4
train_time = 10
train_sample_tree_num = total_num
once_kill_num = 80

train_tree_part_len = int(train_sample_tree_num / train_use_cup_num)
play_tree_part_len = int(play_sample_tree_num / play_use_cup_num)
total_data_len = 80000
train_sample_data_len = 2000

dictionary = None


def show_men_use(show_str):
    print(show_str)
    mem = psutil.virtual_memory()
    print('系统已经使用内存:', (np.float(mem.used) / 1024 / 1024 / 1024), 'GB')
