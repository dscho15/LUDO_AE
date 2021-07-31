import numpy as np

class selector_uniform():

    def __init__(self, prob = 0.5):
        self.prob = prob

    def selection(self, shape):
        
        indices = np.random.uniform(0, 1, shape) > (1-self.prob)

        return indices

class selector_tournament():

    def __init__(self):
        print("hello")

class selector_ftrs():

    def __init__(self, gamma):
        self.gamma = gamma

    def selection(self, fitness):

        cdf = np.zeros(fitness.shape)
        sum = np.sum(fitness)
        
        cdf[0] = fitness[0]/sum
        for i in range(1, len(cdf)):
            cdf[i] = fitness[i]/sum + cdf[i-1]

        current_member = i = 0
        r = np.random.uniform(0, 1/self.gamma)
        indices = np.zeros(self.gamma, dtype=int)

        while current_member < self.gamma:
            while r <= cdf[i]:
                indices[current_member] = i
                r += 1/self.gamma
                current_member += 1
            i += 1
        
        return indices


