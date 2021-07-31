import numpy as np

class mutator_uniform:

    def __init__(self, L, U):
        self.prob = 0.01
        self.L = L
        self.U = U

    def mutate(self, genes):
        
        # Indices 
        indices_to_update = np.random.uniform(0, 1, genes.shape) > (1-self.prob)

        # Uniform mutation
        genes[indices_to_update] = np.random.uniform(self.L, self.U, genes.shape)[indices_to_update]

        # Ensure nobody is above or below 1
        genes[genes > 1] = self.U
        genes[genes < -1] = self.L