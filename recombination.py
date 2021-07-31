import numpy as np


class recombination_one_point_crossover:

    def __init__(self, len_of_genes):
        
        self.len_of_genes = len_of_genes

    def recombine(self, genes, shuffle = True):
        
        # copy genes
        new_genes = np.copy(genes)

        # check if uneven
        size = genes.shape[0]
        if genes.shape[0] % 2 == 1:
            size -= 1

        # shuffle the genes randomly
        indices = np.arange(0, size)
        np.random.shuffle(indices)
        indices = indices.reshape(2,-1)

        # where should we split
        split_chromosome = np.random.randint(1, self.len_of_genes-1, size=(int)(size/2))
        
        # 
        k = 0
        for i in range((int)(size/2)):
            new_genes[k, :split_chromosome[i]] = genes[indices[0, i], :split_chromosome[i]]
            new_genes[k+1, :split_chromosome[i]] = genes[indices[1, i], :split_chromosome[i]]
            new_genes[k, split_chromosome[i]:] = genes[indices[1, i], split_chromosome[i]:]
            new_genes[k+1, split_chromosome[i]:] = genes[indices[0, i], split_chromosome[i]:]
            k += 2

        return new_genes

class recombination_one_point_whole_arith:

    def __init__(self, len_of_genes, alpha = 0.5):

        self.alpha = alpha
        self.len_of_genes = len_of_genes

    def recombine(self, genes, shuffle = True):
        
        # copy genes
        new_genes = np.copy(genes)

        # check if uneven
        size = genes.shape[0]
        if genes.shape[0] % 2 == 1:
            size -= 1

        # shuffle the genes randomly
        indices = np.arange(0, size)
        np.random.shuffle(indices)
        indices = indices.reshape(2,-1)

        # where should we split
        split_chromosome = np.random.randint(1, self.len_of_genes-1, size=(int)(size/2))

        # left or right
        
        # 
        k = 0
        for i in range((int)(size/2)):
            new_genes[k, :split_chromosome[i]] = genes[indices[0, i], :split_chromosome[i]]
            new_genes[k+1, :split_chromosome[i]] = genes[indices[1, i], :split_chromosome[i]]
            new_genes[k, split_chromosome[i]:] = genes[indices[1, i], split_chromosome[i]:]
            new_genes[k+1, split_chromosome[i]:] = genes[indices[0, i], split_chromosome[i]:]
            k += 2

        return new_genes



genes = np.random.uniform(-1, 1, size=(7, 4))
recombination = recombination_one_point_crossover(4)