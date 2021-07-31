import ludopy
import numpy as np

END_ZONE = np.array([54, 55, 56, 57, 58, 59])

def can_leave_home(mv_pcs, player_pcs, dice):
    if dice == 6:
        np.zeros(())
        for i in range(len(mv_pcs)):
            
    if dice == 6:
        return np.where(player_pcs[mv_pcs] == 0)[0]
    return np.zeros((len(mv_pcs), 1))

def can_reach_endzone(mv_pcs, player_pcs, dice):
    
    for i in range(len(mv_pcs)):
        if np.any(player_pcs[mv_pcs[i]] + dice == END_ZONE)


print(can_reach_endzone(np.array([0, 1]), np.array([50, 51, 0, 0]), 6))


def get_action(self, chromosome_idx, player_pcs, mv_pcs, enemy_pcs, dice):

    

    # If no actions are available choose -1.
    if(len(mv_pcs) == 0):
        return -1

    # If only one piece is available, then that is the piece to be moved.
    if(len(mv_pcs) == 1):
        return mv_pcs[0]

    # Compute a boolean state array, the representation is on top [0 0 0 0 0 0 0 0 0]
    state = np.zeros((len(mv_pcs), self.number_of_genes))

    # If some of the pieces can be moved, e.g. not home - it is an index
    idx_moveable = np.where(player_pcs[mv_pcs] > 0)[0]

    # And make a variable that takes the dice roll, this is a dice roll.
    pos_after_dice = player_pcs[idx_moveable] + dice

    # We want to iterate for each index
    for i in range(len(idx_moveable)):

        if (pos_after_dice[i] == GOAL) == True:
            state[idx_moveable[i], REACH_GOAL] = 1

        if np.any(pos_after_dice[i] == GLOB_INDEXS) == True:
            state[idx_moveable[i], REACH_GLOB] = 1

        if np.any(pos_after_dice[i] == STAR_INDEXS) == True:
            state[idx_moveable[i], REACH_STAR] = 1
        
        # The normalized distance it has moved
        state[idx_moveable[i], NORM_DISTANCE] = (player_pcs[idx_moveable[i]] ) / 56

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
            
            # What is the probability that enemy hits the piece home?
            
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
