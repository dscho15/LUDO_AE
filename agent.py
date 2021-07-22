import ludopy
import numpy as np
import itertools

#   STATE REPRESENTATION:
#   0. Token reaches goal                                           - done
#   1. Token knocks home enemy                                      - done
#   2. Token lands on globe                                         - done
#   3. Token lands on a star                                        - done
#   4. Token to leave home square                                   - done
#   5. Token is safe                                                - done
#   6. There are more enemies infront of the token after it moves   - working on it
#   7. There are more enemies behind the token after it moves       - working on it

REACH_GOAL = 0
KNOCK_ENEM = 1
REACH_GLOB = 2
REACH_STAR = 3
LEAVE_HOME = 4
STAY_SAFE = 5

# Is it possible to move to a goal, globe, star or safe.
HOME_INDEX = 0
GOAL_INDEX = 59
GLOB_INDEXS = [9, 22, 35, 48]
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]
SAFE_INDEXS = [1, 9, 22, 35, 48, 53, 54, 55, 56, 57, 58]


class Population:

    def __init__(self, size):
        self.mutations = 0
        self.sigma = 0.01
        self.size = size
        self.arr = np.random.uniform(-1, 1, size)

    def crossover_onepoint_avg(self, prob=0.65):

        # Figure out which chromosomes to replace
        self.who_to_update = np.random.uniform(0, 1, self.size[0]) > (1-prob)

        # Pair

    def mutate(self, prob=0.01, tsh=0.001):

        # Figure out which genes to replace
        self.who_to_update = np.random.uniform(0, 1, self.size) > (1-prob)

        # Calculate tau based on mutations
        self.mutations += 1
        tau = 1/self.mutations

        # Is smaller than threshhold
        if self.sigma < tsh:
            self.sigma = tsh
        else:
            self.sigma = self.sigma * np.exp(np.random.normal(0, 1) * tau)

        # Uncorrelated Mutation with One Step Size
        self.arr[self.who_to_update] = self.arr[self.who_to_update] + \
            self.sigma * np.random.normal(0, 1, self.size)[self.who_to_update]

        # Ensure nobody is above or below 1
        self.arr[self.arr > 1] = 1.0
        self.arr[self.arr < -1] = -1.0

        print(self.sigma)
        print(np.count_nonzero(self.who_to_update == True))


class Agent:

    def __init__(self, env, size=np.array([100, 8])):
        self.g = env.Game()
        self.pop_size = size
        self.population = Population(size)

    def get_action(self, chromosome_idx, player_pcs, mv_pcs, enemy_pcs, dice):

        # If no actions are available choose -1.
        if(len(mv_pcs) == 0):
            return -1

        # If only one piece is available, then that is the piece to be moved.
        if(len(mv_pcs) == 1):
            return mv_pcs[0]

        # Compute a boolean state array, the representation is on top [0 0 0 0 0 0 0 0 0]
        state = np.zeros(shape=(len(mv_pcs), 8))

        # Only if the dice is 6.
        if dice == 6:

            # If some of the pieces are at home, then localize which we are talking about and flip the bits
            arr_home = np.where(player_pcs[mv_pcs] == HOME_INDEX)

            # Update their chromosomes
            for i in range(len(arr_home[0])):
                state[arr_home[0][i], LEAVE_HOME] = 1

        # If some of the pieces can be moved, e.g. not home - it is an index
        arr_not_home_idx = np.where(player_pcs[mv_pcs] > 0)[0]

        # And make a variable that takes the dice roll, this is a dice roll.
        arr_not_home_dice = player_pcs[arr_not_home_idx] + dice

        # We want to iterate for each index
        for i in range(len(arr_not_home_idx)):

            # Can we move to goal?
            if np.any(arr_not_home_dice[i] == GOAL_INDEX) == True:
                state[arr_not_home_idx[i], REACH_GOAL] = 1

            # Can we knock an enemy? Figure out where they are positioned relative to their own home, offsets of 13, 26, 39
            move_to_enemy_pov_dice = list(itertools.chain(*ludopy.player.enemy_pos_at_pos(arr_not_home_dice[i])))

            # Iterate through enemy_pcs
            for j in range(len(enemy_pcs)):

                # If enemy_peaces are at same place as move_to_enemy_pov_dice, and there is only one, then it is considered good.
                if np.count_nonzero(np.where(enemy_pcs[j] == move_to_enemy_pov_dice[j])) == 1:
                    state[arr_not_home_idx[i], KNOCK_ENEM] = 1
                    break

            # Can we move to one of the globe index
            if np.any(arr_not_home_dice[i] == GLOB_INDEXS) == True:
                state[arr_not_home_idx[i], REACH_GLOB] = 1

            # Can we move to one of the star index
            if np.any(arr_not_home_dice[i] == STAR_INDEXS) == True:
                state[arr_not_home_idx[i], REACH_STAR] = 1

            # Can we move to a safe index
            if np.any(player_pcs[arr_not_home_idx[i]] == SAFE_INDEXS) == True:
                state[arr_not_home_idx[i], STAY_SAFE] = 1

            # More enemies are infront when you move.
            # Figure out how many tokens that are infront of the token before the movement, and then after the movement.
            move_to_enemy_pov = np.array(list(itertools.chain(*ludopy.player.enemy_pos_at_pos(player_pcs[arr_not_home_idx[i]]))))
            
            # More enemies are behind when you move
            for j in range(len(enemy_pcs)):

                # Remove
                enemy_pov_on_player = enemy_pcs[j][np.where(np.logical_and(enemy_pcs[j] > 0, enemy_pcs[j] <= 53))[0]]

                if len(enemy_pov_on_player) == 0:
                    continue
                else:
                    
                    # Add or subtract 6
                    move_to_enemy_pov_removed_6 = move_to_enemy_pov - 6
                    move_to_enemy_pov_added_6 = move_to_enemy_pov + 6

                    # Make sure the limit holds
                    #limits_lower = move_to_enemy_pov_removed_6[move_to_enemy_pov_removed_6 >= 1 ]
                    #limits_upper = move_to_enemy_pov_added_6[move_to_enemy_pov_added_6 <= 53 ]

                    print(move_to_enemy_pov)

                    print(player_pcs[arr_not_home_idx[i]], move_to_enemy_pov_removed_6, move_to_enemy_pov_added_6)
                    


                # Minus the enemy value with 6m
                #six_steps_behind -= 6

                #if np.count_nonzero(np.where(enemy_pcs[j] == move_to_enemy_pov_dice[j])) == 1:

        #exit()

        # now compute what action to take
        arr = mv_pcs[np.argmax(
            state @ np.expand_dims(self.population.arr[chromosome_idx], axis=1))]

        # Return action
        return arr

    def evaluate_players(self, iterations):

        # holds the winrate
        win_rate = np.zeros((self.pop_size[0], 1))

        # There should be a round for each chromosome
        for i in range(self.pop_size[0]):

            for j in range(iterations):

                win = False
                rounds = 0

                while not win:

                    # Get a non-preprocessed state by the player_pc and enemy_pc
                    (dice, mv_pcs, player_pcs, enemy_pcs, player_win,
                     win), player_i = self.g.get_observation()

                    # Pick a random choice, e.g. train vs random noobs.
                    if player_i != 0:
                        # Move pieces by a uniform distribution
                        if len(mv_pcs) > 0:
                            piece_to_move = np.random.choice(mv_pcs)
                        else:
                            piece_to_move = -1
                    else:
                        # Select action if u are player 0
                        piece_to_move = self.get_action(
                            i, player_pcs, mv_pcs, enemy_pcs, dice)

                    _, _, _, _, _, win = self.g.answer_observation(
                        piece_to_move)

                    # Immediately break if a player has won
                    if win:
                        break

                    rounds += 1

                # If player zero wins, then update winrate
                if self.g.get_winner_of_game() == 0:
                    win_rate[i] += 1

                self.g.reset()

            print("Evaluating chromosome: ", i)

        return win_rate/iterations

    # def test(self):
    #     for i in range(20000):
    #         self.population.un_mutate()
    #         self.population.one_point_crossover()


def main():
    agent = Agent(ludopy)
    win_rate = agent.evaluate_players(100)
    print(win_rate)
    # agent.test()


if __name__ == '__main__':
    main()
