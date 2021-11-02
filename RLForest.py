import numpy as np
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
import copy
from collections import deque
import random
import joblib
from joblib.parallel import Parallel, delayed
import gym


class Tree(object):
    def __init__(self, data, target, max_depth, state_len, use_feature_num):
        self.max_depth = max_depth
        self.state_len = state_len
        self.use_feature_num = use_feature_num
        self.reality_tree = DecisionTreeRegressor(
            max_depth=self.max_depth
        )
        self.use_feature = np.arange(self.state_len)
        np.random.shuffle(self.use_feature)
        self.use_feature = self.use_feature[:self.use_feature_num]
        data = np.array(data)
        target = np.array(target)

        self.reality_tree.fit(data[:, self.use_feature], target)
        self.visit = 0
        self.evaluate_value = 0
        self.difference_value = 0

    def get_forecast(self, data):
        data = np.array(data)
        data.reshape(-1, self.state_len)
        return self.reality_tree.predict(data[:, self.use_feature])


class Forest(object):
    def __init__(
            self,
            n_estimators=200,
            state_len=5,
            max_depth=3,
            use_feature_num=3,
    ):
        self.trees = []
        self.n_estimators = np.int(n_estimators)
        self.state_len = state_len
        self.max_depth = max_depth
        self.use_feature_num = use_feature_num

    def init_random_forest(self, random_col=5000, sample_len=100, class_kind=5):
        tmp_forest = []
        init_data = np.random.randint(2, size=(random_col, self.state_len))
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
                    self.state_len,
                    self.use_feature_num
                )
            )
        self.trees = tmp_forest

    def forest_predict(self, data):
        tmp_sum = []
        for i_tree in range(self.n_estimators):
            tmp_sum.append(self.trees[i_tree].get_forecast(data))
        return np.mean(tmp_sum, axis=0)

    def train(self, data, target, kill):
        # show_men_use()
        max_play = -1
        tmp_sum = [tmp_tree.get_forecast(data) for tmp_tree in self.trees]

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

        base_evaluate = np.zeros(self.n_estimators)  # 去掉這一颗树的分數(越大越好)
        base_bias = np.zeros(self.n_estimators)  # 树和目標的距離(越小越好)
        base_difference = np.zeros(self.n_estimators)  # 树和其余树的距離(越大越好)

        evaluate_history = np.zeros(self.n_estimators)
        e_total_evaluate = np.zeros(self.n_estimators)
        d_total_evaluate = np.zeros(self.n_estimators)

        # 用相对排名(不用绝对数据)代表分数
        for i_tree in range(self.n_estimators):
            reject_ans = (total_ans - tmp_sum[i_tree, :]) / (self.n_estimators - 1)
            base_evaluate[i_tree] = np.mean((reject_ans - target) ** 2)  # 丢掉他之后离target越进，越说明拖后腿 所以越大越好
            base_bias[i_tree] = np.mean((tmp_sum[i_tree, :] - target) ** 2)  # 离中心近好 所以越小越好
            base_difference[i_tree] = np.mean((tmp_sum[i_tree, :] - mean_ans) ** 2)  # 离中心远好 所以越大越好

        mean_bias = np.mean(base_bias)  # 用作分数权重
        mean_difference = np.mean(base_difference)  # 用作分数权重

        base_evaluate = base_evaluate - np.mean(base_evaluate)
        base_evaluate /= np.std(base_evaluate)  # 小的在前(好的在後)
        # base_evaluate /= 3

        base_bias = base_bias - np.mean(base_bias)
        base_bias /= np.std(base_bias)  # 小的在前(好的在前)
        # base_bias /= 3

        base_difference = base_difference - np.mean(base_difference)
        base_difference /= np.std(base_difference)  # 小的在前(好的在後)
        # base_difference /= 3

        e_total_evaluate += (1 + 1) * base_evaluate
        d_total_evaluate -= 1 * base_bias
        d_total_evaluate += 1 * base_difference

        for i_tree in range(self.n_estimators):
            self.trees[i_tree].evaluate_value = (self.trees[i_tree].evaluate_value * self.trees[
                i_tree].visit + e_total_evaluate[i_tree]) / (self.trees[i_tree].visit + 1)

            self.trees[i_tree].difference_value = (self.trees[i_tree].difference_value * self.trees[
                i_tree].visit + d_total_evaluate[i_tree]) / (self.trees[i_tree].visit + 1)

            self.trees[i_tree].visit += 1
            evaluate_history[i_tree] = self.trees[i_tree].difference_value + self.trees[
                i_tree].evaluate_value + 0.1 * np.sqrt(np.log(max_play) / (1 + max_play - self.trees[i_tree].visit))

        print('误差:', total_bias)
        print('平均值分数:', np.mean(evaluate_history))
        print('target平均距離:', mean_bias)
        print('difference_base平均距離:', mean_difference)
        print('target平均距離和difference的差距:', mean_bias - mean_difference)

        # 删除差的树
        idx = np.arange(self.n_estimators)
        good_idx = idx[evaluate_history.argsort()][kill:]
        new_trees = [self.trees[i_tree] for i_tree in good_idx]
        self.trees = new_trees

        # 计算剩下的树和目标的差距
        tmp_sum = tmp_sum[good_idx, :]
        need_fix_bias = (target - np.mean(tmp_sum, axis=0)) * (self.n_estimators - kill)
        # 计算所需修正
        data = np.reshape(data, (-1, self.state_len))
        fix_bias = need_fix_bias / kill
        target_change = target + fix_bias
        # 补全剩下的树
        for i_tree in range(kill):
            self.trees.append(
                Tree(
                    data,
                    target_change,
                    self.max_depth,
                    self.state_len,
                    self.use_feature_num
                )
            )

        return total_bias

    def save_model(self, loop, local_dir):
        joblib.dump(self.trees, local_dir + '(' + str(loop) + ').pkl')

    def read_model(self, loop, local_dir, is_refresh):
        tmp_dic = local_dir + '(' + str(loop) + ').pkl'
        print(tmp_dic)
        self.trees = joblib.load(tmp_dic)
        if is_refresh:
            for i_tree in range(self.n_estimators):
                self.trees[i_tree].visit = 0
                self.trees[i_tree].evaluate_value = 0
                self.trees[i_tree].difference_value = 0


class ForestAgent(object):
    def __init__(self, **kwargs):
        self.gamma = 0.9
        self.batch_size = 128
        self.capacity = 1000
        self.decay = 200
        self.epsilon_high = 0.9
        self.epsilon_low = 0.05
        self.lr = 0.0002
        self.action_space_dim = 0
        self.state_space_dim = 0
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.buffer = []
        self.steps = 0

        self.forest = Forest()
        self.forest.init_random_forest()

    def act(self, obs):
        self.steps += 1
        epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * (np.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsilon:
            action = random.randrange(self.action_space_dim)
        else:
            tmp_obs = [list(obs) + [0], list(obs) + [1]]
            action_q = self.forest.forest_predict(tmp_obs)
            action_q += (np.random.randn(2) * 0.05)
            action = np.int(action_q[1] > action_q[0])
        return action

    def put(self, *transition):
        while len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = np.array(s0)
        a0 = np.array(a0)
        r1 = np.array(r1)
        s1 = np.array(s1)
        q_value0 = self.forest.forest_predict(np.insert(s1, 4, values=np.zeros(self.batch_size), axis=1))
        q_value1 = self.forest.forest_predict(np.insert(s1, 4, values=np.ones(self.batch_size), axis=1))
        q_value = np.max([q_value0, q_value1], axis=0)

        y_true = r1 + self.gamma * q_value
        y_pred = self.forest.forest_predict(np.insert(s0, 4, values=a0, axis=1))
        y_train = y_true *1 + y_pred * 0
        self.forest.train(np.insert(s0, 4, values=a0, axis=1), y_train, 10)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    params = {
        'gamma'           : 0.8,
        'epsilon_high'    : 0.9,
        'epsilon_low'     : 0.05,
        'decay'           : 200,
        'lr'              : 0.001,
        'capacity'        : 10000,
        'batch_size'      : 32,
        'state_space_dim' : 5,
        'action_space_dim': 2
    }
    agent = ForestAgent(**params)
    score = []
    mean = []

    for episode in range(2000):
        obs = env.reset()
        total_reward = 1
        while True:
            # env.render()
            act = agent.act(obs)
            new_obs, reward, done, _ = env.step(act)
            if done:
                reward = -1
            agent.put(obs, act, reward, new_obs)

            if done:
                break

            total_reward += reward
            obs = new_obs

        print(episode, '轮完：')
        print(total_reward)
        agent.learn()
        score.append(total_reward)
        mean.append(sum(score[-100:]) / 100)
