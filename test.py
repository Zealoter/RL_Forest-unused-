from RLForest import Forest
import numpy as np

# np.random.seed(0)
# forest = Forest()
# forest.init_random_forest()
# data = np.random.rand(5, 4)
# target = np.sum(data, axis=1)
# print(data)
# print(target)
# guess = forest.forest_predict(data)
# print(guess)
# forest.train(data, target, 20)
# guess = forest.forest_predict(data)
# print(guess)
# a = [np.array([1, 2, 3]), np.array([2, 3, 4])]
# print(np.max(a, axis=0))
s1 = np.array([[1, 2], [2, 3]])
s1 = np.insert(s1, 2, values=np.zeros(2), axis=1)
print(s1)
