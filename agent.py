import ludopy
import numpy as np
import itertools
from multiprocessing import Process, Array

#   STATE REPRESENTATION:
#   0. Token reaches goal zone
#   1. Can knock home enemy
#   2. Can reach globe
#   3. Can reach star
#   4. Leaves the home
#   5. Can stay safe
#   6. Is in the goal zone
#   7. There were more enemies behind before, than after it moved
#   8. There were more enemies infront before, than after it moved
#   9.  Is first -
#   10. Is second -
#   11. Is third -
#   12. Is fourth -

REACH_GOAL = 0
KNOCK_ENEM = 1
REACH_GLOB = 2
REACH_STAR = 3
LEAVE_HOME = 4
STAY_SAFE = 5
GOAL_ZONE = 6
LOW_COUNT = 7
UPP_COUNT = 8

# Is it possible to move to a goal, globe, star or safe.
HOME_INDEX = 0
GOAL = 59
GLOB_INDEXS = [9, 22, 35, 48]
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]
SAFE_INDEXS = [1, 9, 22, 35, 48, 53, 54, 55, 56, 57, 58, 59]
GOAL_INDEXS = [54, 55, 56, 57, 58, 59]
ONE_ROUND = 53


class Population:

    def __init__(self, size):

        self.mutations = 0
        self.sigma = 0.01
        self.size = size
        self.arr = np.random.uniform(-1, 1, size)

    def recombination_avg(self):
    
        # What parents should I choose?
        np.random.shuffle(self.parents)

        # Don't touch the last if it is uneven
        if self.parents.shape[0] % 2 == 0:
            par_size = self.parents.shape[0]
        else:
            par_size = self.parents.shape[0] - 1

        # Should I pick left or right
        left = np.random.randint(0, 2, size=(int)(par_size/2))

        # Figure out which chromosomes to replace
        idx_update = np.random.randint(1, self.size[1]-1, size=(int)(par_size/2))

        # What chromosomes should I update?
        self.parents_temp = np.copy(np.split(self.parents[0:par_size], 2, axis=0))

        # k == 0, means
        k = 0
        for i in range((int)(par_size/2)):
            
            # if left is true, then insert them in the left side
            if left[i] == 1:
                self.parents[k, :idx_update[i]] = self.parents_temp[0][i, :idx_update[i]]
                self.parents[k+1, :idx_update[i]] = self.parents_temp[1][i, :idx_update[i]]
                self.parents[k, idx_update[i]:] = (self.parents_temp[0][i, idx_update[i]:] + self.parents_temp[1][i, idx_update[i]:])/2
                self.parents[k+1, idx_update[i]:] = (self.parents_temp[0][i, idx_update[i]:] + self.parents_temp[1][i, idx_update[i]:])/2
            else:
                # else insert the genes in right side
                self.parents[k, idx_update[i]:] = self.parents_temp[0][i, idx_update[i]:]
                self.parents[k+1, idx_update[i]:] = self.parents_temp[1][i, idx_update[i]:]
                self.parents[k, :idx_update[i]] = (self.parents_temp[0][i, :idx_update[i]] + self.parents_temp[1][i, :idx_update[i]])/2
                self.parents[k+1, :idx_update[i]] = (self.parents_temp[0][i, :idx_update[i]] + self.parents_temp[1][i, :idx_update[i]])/2

            k += 2

    def recombination(self):
        
        # What parents should I choose?
        np.random.shuffle(self.parents)

        # Don't touch the last if it is uneven
        if self.parents.shape[0] % 2 == 0:
            par_size = self.parents.shape[0]
        else:
            par_size = self.parents.shape[0] - 1

        # Should I pick left or right
        left = np.random.randint(0, 2, size=(int)(par_size/2))

        # Figure out which chromosomes to replace
        idx_update = np.random.randint(1, self.size[1]-1, size=(int)(par_size/2))

        # What chromosomes should I update?
        self.parents_temp = np.copy(np.split(self.parents[0:par_size], 2, axis=0))

        # k == 0, means
        k = 0
        for i in range((int)(par_size/2)):
            
            # if left is true, then insert them in the left side
            if left[i] == 1:
                self.parents[k, :idx_update[i]] = self.parents_temp[0][i, :idx_update[i]]
                self.parents[k+1, :idx_update[i]] = self.parents_temp[1][i, :idx_update[i]]
                self.parents[k, idx_update[i]:] = self.parents_temp[1][i, idx_update[i]:]
                self.parents[k+1, idx_update[i]:] = self.parents_temp[0][i, idx_update[i]:]
            else:
                # else insert the genes in right side
                self.parents[k, idx_update[i]:] = self.parents_temp[0][i, idx_update[i]:]
                self.parents[k+1, idx_update[i]:] = self.parents_temp[1][i, idx_update[i]:]
                self.parents[k, :idx_update[i]] = self.parents_temp[1][i, :idx_update[i]]
                self.parents[k+1, :idx_update[i]] = self.parents_temp[0][i, :idx_update[i]]

            k += 2

    def mutate(self, prob=0.02):
    
        # Figure out which genes to replace
        who_to_update = np.random.uniform(0, 1, self.parents.shape) > (1-prob)

        # Uncorrelated Mutation with One Step Size
        self.parents[who_to_update] = np.random.uniform(-1, 1, self.parents.shape)[who_to_update]

        # Ensure nobody is above or below 1
        self.parents[self.parents > 1] = 1.0
        self.parents[self.parents < -1] = -1.0

    def parent_selection(self, prob=0.50):

        # Determine who is gonna be the parents
        who_to_update = np.where(np.random.uniform(0, 1, self.size[0]) > (1-prob))[0]

        # Those who are higher than the probability are chosen as parents, the indices are now taken.
        self.parents = np.copy(self.arr[who_to_update])

    def update_genes(self):
        self.arr = np.vstack((self.parents, self.arr))
        print(len(self.arr))

    def __len__(self):
        return self.arr.shape[0]


    def survivor_selection(self, fitness):
    
        # Stochastic Universal Sampling [SUS]
        a = np.zeros((self.arr.shape[0], 1))
        sum = np.sum(fitness)
        
        # Calculate cdf
        for i in range(0, len(a)):
            if[i] == 0:
                a[i] = fitness[i]/sum
            else:
                a[i] = fitness[i]/sum + a[i-1]

        # We wish to select gamma members, aka 100 members
        gamma = self.size[0]
        current_member = i = 0
        r = np.random.uniform(0, 1/gamma)
        mating_pool = np.zeros(gamma, dtype=int)

        # SUS Algorithm p. 84
        while current_member < gamma:
            while r <= a[i]:
                mating_pool[current_member] = i
                r += 1/gamma
                current_member += 1
            i += 1
        
        # Update the array
        self.arr = np.copy(self.arr[mating_pool])
        return mating_pool

class Agent:

    def __init__(self, env, size=np.array([100, 9])):

        self.g = env
        self.number_of_genes = size[1]
        self.population = Population(size)

    def get_action(self, chromosome_idx, player_pcs, mv_pcs, enemy_pcs, dice):

        # If no actions are available choose -1.
        if(len(mv_pcs) == 0):
            return -1

        # If only one piece is available, then that is the piece to be moved.
        if(len(mv_pcs) == 1):
            return mv_pcs[0]

        # Compute a boolean state array, the representation is on top [0 0 0 0 0 0 0 0 0]
        state = np.zeros((len(mv_pcs), self.number_of_genes))

        # Only if the dice is 6.
        if dice == 6:

            # If some of the pieces are at home, then localize which we are talking about and flip the bits
            arr_home = np.where(player_pcs[mv_pcs] == HOME_INDEX)

            # Update their chromosomes
            for i in range(len(arr_home[0])):
                state[arr_home[0][i], LEAVE_HOME] = 1

        # If some of the pieces can be moved, e.g. not home - it is an index
        idx_moveable = np.where(player_pcs[mv_pcs] > 0)[0]

        # And make a variable that takes the dice roll, this is a dice roll.
        pos_after_dice = player_pcs[idx_moveable] + dice

        # We want to iterate for each index
        for i in range(len(idx_moveable)):

            # Can we move to goal?
            if (pos_after_dice[i] == GOAL) == True:
                state[idx_moveable[i], REACH_GOAL] = 1

            # Can we move to goal zone?
            if player_pcs[idx_moveable][i] <= ONE_ROUND:
                if np.any(pos_after_dice[i] == GOAL_INDEXS) == True:
                    state[idx_moveable[i], GOAL_ZONE] = 1

            # Can we move to one of the globe indexs?
            if np.any(pos_after_dice[i] == GLOB_INDEXS) == True:
                state[idx_moveable[i], REACH_GLOB] = 1

            # Can we move to one of the star indexs?
            if np.any(pos_after_dice[i] == STAR_INDEXS) == True:
                state[idx_moveable[i], REACH_STAR] = 1

            # Can we move to a safe index - this method does not include friendly players, as they could potentially move?
            if np.any(player_pcs[idx_moveable[i]] == SAFE_INDEXS) == True:
                state[idx_moveable[i], STAY_SAFE] = 1

            # Only make sense to compute if player is less than 53.
            if player_pcs[idx_moveable[i]] < 53:

                # Can we knock an enemy? Figure out where they are positioned relative to their own home, offsets of 13, 26, 39
                pos_enemy_pov = np.array(list(itertools.chain(*ludopy.player.enemy_pos_at_pos(player_pcs[idx_moveable[i]]))))
                pos_enemy_pov_dice = np.array(list(itertools.chain(*ludopy.player.enemy_pos_at_pos(pos_after_dice[i]))))

                # Iterate through enemy_pcs
                for j in range(len(enemy_pcs)):

                    # If enemy_peaces are at same place as pos_enemy_pov_dice, and there is only one, then the piece can knock
                    # enemy player home
                    if np.count_nonzero(np.where(enemy_pcs[j] == pos_enemy_pov_dice[j])) == 1:
                        state[idx_moveable[i], KNOCK_ENEM] = 1
                        break
                
                # variables to count enemy pieces before and after
                lower_counts_before, lower_counts_after = 0, 0
                upper_counts_before, upper_counts_after = 0, 0

                # more enemies are behind when you move
                for j in range(len(enemy_pcs)):

                    # keep all over zero and all those under or equal to 53.
                    enemy_pcs_reduced = enemy_pcs[j][np.where(np.logical_and(enemy_pcs[j] > 0, enemy_pcs[j] <= 53))[0]]

                    # if empty
                    if len(enemy_pcs_reduced) == 0:
                        continue
                    else:

                        # Add or subtract 6
                        lower_lim = pos_enemy_pov[j] - 6
                        if lower_lim < 1:
                            lower_lim = 1
                        upper_lim = pos_enemy_pov[j] + 6

                        # Before
                        lower_counts_before += np.count_nonzero(np.logical_and(
                            lower_lim <= enemy_pcs_reduced, pos_enemy_pov[j] > enemy_pcs_reduced))
                        upper_counts_before += np.count_nonzero(np.logical_and(
                            upper_lim >= enemy_pcs_reduced, pos_enemy_pov[j] < enemy_pcs_reduced))

                        # Add or subtract after potential move.
                        lower_lim = pos_enemy_pov_dice[j] - 6
                        if lower_lim < 1:
                            lower_lim = 1
                        upper_lim = pos_enemy_pov_dice[j] + 6

                        # After
                        lower_counts_after += np.count_nonzero(np.logical_and(
                            lower_lim <= enemy_pcs_reduced, pos_enemy_pov_dice[j] > enemy_pcs_reduced))
                        upper_counts_after += np.count_nonzero(np.logical_and(
                            upper_lim >= enemy_pcs_reduced, pos_enemy_pov_dice[j] < enemy_pcs_reduced))

                if lower_counts_before > lower_counts_after:
                    state[idx_moveable[i], LOW_COUNT] = 1

                if upper_counts_before > upper_counts_after:
                    state[idx_moveable[i], UPP_COUNT] = 1

        # now compute what action to take
        arr = mv_pcs[np.argmax(state @ np.expand_dims(self.population.arr[chromosome_idx], axis=1))]

        # return that action
        return arr

    def get_fitness(self, iterations):

        # holds the winrate
        win_rate = Array('i', np.zeros(len(self.population), dtype=int))

        # create the linespace
        x = np.linspace(0, len(self.population), num=17, dtype=int)
        processes = []

        # get the different startvalues
        for i in range(len(x)-1):
            processes.append(Process(target=self.simulate_games, args=(x[i], x[i+1], iterations, win_rate)))
            processes[i].start()

        # started 
        for i in range(len(processes)):
            processes[i].join()

        self.win_rate = np.array(win_rate[:], dtype=float)/iterations

    def simulate_games(self, start, end, iterations, win_rate):
        
        game = self.g.Game()

        for i in range(start, end): 

            for j in range(iterations):

                win = False
                rounds = 0

                while not win:

                    # Get a non-preprocessed state by the player_pc and enemy_pc
                    (dice, mv_pcs, player_pcs, enemy_pcs, _, win), player_i = game.get_observation()

                    # Pick a random choice, e.g. train vs random noobs.
                    if player_i != 0:
                        if len(mv_pcs) > 0:
                            piece_to_move = np.random.choice(mv_pcs)
                        else:
                            piece_to_move = -1
                    else:
                        piece_to_move = self.get_action(i, player_pcs, mv_pcs, enemy_pcs, dice)

                    _, _, _, _, _, win = game.answer_observation(piece_to_move)

                    # Immediately break if a player has won
                    if win:
                        break

                    rounds += 1

                # If player zero wins, then update winrate
                if game.get_winner_of_game() == 0:
                    win_rate[i] += 1

                game.reset()

    def train_agent(self, n, iterations):

        ws = np.zeros((n, 200))

        for i in range(n):

            print("Parent selection...")

            self.population.parent_selection()

            print("Recombination...")

            self.population.recombination()

            print("Mutate...")

            self.population.mutate()

            print("Update genes...")

            self.population.update_genes()

            print("Compute fitness")

            self.get_fitness(iterations)

            print("Survivor selection")

            mating_pool = self.population.survivor_selection(self.win_rate)

            ws[i, :] = self.win_rate[mating_pool]

        np.savetxt('win_rate.out', ws, delimiter=',')

        np.savetxt('genes.out', self.population.arr, delimiter=',')


def main():

    agent = Agent(ludopy, np.array([200, 9]))

    agent.train_agent(50, 500)


if __name__ == '__main__':
    main()
