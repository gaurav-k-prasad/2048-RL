import copy
import random
import numpy as np

INVALID_ACTION_PENALTY = -8


class Game2048:
    def __init__(self, prob2: float = 0.9, learning_mode="ann", board=None):
        self.prob2 = prob2
        self.prob4 = 1 - prob2
        self.learning_mode = learning_mode
        self.user_board = None
        if board:
            self.user_board = board.copy()

        self.max_tile_value = 0
        self.new_game()
        self.actions_map = [
            (self.move_up, self.is_possible_up),
            (self.move_down, self.is_possible_down),
            (self.move_left, self.is_possible_left),
            (self.move_right, self.is_possible_right),
        ]
        self.actions = ['U', 'D', 'L', 'R']

        self.merge_tile_bonus = 1.2
        self.empty_count_bonus = 0.1
        self.cornered_bonus = 0.8
        self.smoothness_bonus = 0.07

    def is_game_over(self) -> bool:
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return False
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return False
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return False

        return True

    def find_empty_slots(self) -> list[tuple[int, int]]:
        res = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    res.append((i, j))
        return res

    def place_new_block(self) -> int:
        empty_slots = self.find_empty_slots()
        i, j = random.choice(empty_slots)
        self.board[i][j] = 2 if random.random() < self.prob2 else 4
        return self.board[i][j]

    def is_possible_up(self) -> bool:
        for j in range(4):
            is_gap = False
            for i in range(4):
                if (
                    i < 3
                    and self.board[i][j]
                    and self.board[i][j] == self.board[i + 1][j]
                ):
                    return True

                if is_gap and self.board[i][j] != 0:
                    return True
                if self.board[i][j] == 0:
                    is_gap = True

        return False

    def is_possible_down(self) -> bool:
        for j in range(4):
            is_gap = False
            for i in range(3, -1, -1):
                if (
                    i < 3
                    and self.board[i][j]
                    and self.board[i][j] == self.board[i + 1][j]
                ):
                    return True

                if is_gap and self.board[i][j] != 0:
                    return True
                if self.board[i][j] == 0:
                    is_gap = True

        return False

    def is_possible_left(self) -> bool:
        for i in range(4):
            is_gap = False
            for j in range(4):
                if (
                    j < 3
                    and self.board[i][j]
                    and self.board[i][j] == self.board[i][j + 1]
                ):
                    return True

                if is_gap and self.board[i][j] != 0:
                    return True
                if self.board[i][j] == 0:
                    is_gap = True

        return False

    def is_possible_right(self) -> bool:
        for i in range(4):
            is_gap = False
            for j in range(3, -1, -1):
                if (
                    j < 3
                    and self.board[i][j]
                    and self.board[i][j] == self.board[i][j + 1]
                ):
                    return True

                if is_gap and self.board[i][j] != 0:
                    return True
                if self.board[i][j] == 0:
                    is_gap = True

        return False

    def move_up(self) -> float:
        merged_tiles_log_sum = 0

        for j in range(4):
            idx = 0
            prev = -1
            i = 0
            while i < 4:
                if self.board[i][j] != 0:
                    if prev == -1:
                        prev = self.board[i][j]

                        # shifting up
                        self.board[i][j] = 0
                        self.board[idx][j] = prev
                        idx += 1
                    else:
                        if prev == self.board[i][j]:
                            self.board[idx - 1][j] += self.board[i][j]
                            merged_tiles_log_sum += (
                                self.board[idx - 1][j].bit_length() - 1
                            )
                            self.max_tile_value = max(
                                self.board[idx - 1][j], self.max_tile_value
                            )
                            self.board[i][j] = 0
                            prev = -1
                        else:
                            prev = self.board[i][j]

                            # shifting up
                            self.board[i][j] = 0
                            self.board[idx][j] = prev
                            idx += 1
                i += 1

        return merged_tiles_log_sum

    def move_down(self) -> float:
        merged_tiles_log_sum = 0
        for j in range(4):
            idx = 3
            prev = -1
            i = 3
            while i >= 0:
                if self.board[i][j] != 0:
                    if prev == -1:
                        prev = self.board[i][j]

                        # shifting down
                        self.board[i][j] = 0
                        self.board[idx][j] = prev
                        idx -= 1
                    else:
                        if prev == self.board[i][j]:
                            self.board[idx + 1][j] += self.board[i][j]
                            merged_tiles_log_sum += (
                                self.board[idx + 1][j].bit_length() - 1
                            )
                            self.max_tile_value = max(
                                self.board[idx + 1][j], self.max_tile_value
                            )
                            self.board[i][j] = 0
                            prev = -1
                        else:
                            prev = self.board[i][j]

                            # shifting down
                            self.board[i][j] = 0
                            self.board[idx][j] = prev
                            idx -= 1
                i -= 1

        return merged_tiles_log_sum

    def move_left(self) -> float:
        merged_tiles_log_sum = 0
        for i in range(4):
            idx = 0
            prev = -1
            j = 0
            while j < 4:
                if self.board[i][j] != 0:
                    if prev == -1:
                        prev = self.board[i][j]

                        # shifting left
                        self.board[i][j] = 0
                        self.board[i][idx] = prev
                        idx += 1
                    else:
                        if prev == self.board[i][j]:
                            self.board[i][idx - 1] += self.board[i][j]
                            merged_tiles_log_sum += (
                                self.board[i][idx - 1].bit_length() - 1
                            )
                            self.max_tile_value = max(
                                self.board[i][idx - 1], self.max_tile_value
                            )
                            self.board[i][j] = 0
                            prev = -1
                        else:
                            prev = self.board[i][j]

                            # shifting left
                            self.board[i][j] = 0
                            self.board[i][idx] = prev
                            idx += 1
                j += 1
        return merged_tiles_log_sum

    def move_right(self) -> float:
        merged_tiles_log_sum = 0
        for i in range(4):
            idx = 3
            prev = -1
            j = 3
            while j >= 0:
                if self.board[i][j] != 0:
                    if prev == -1:
                        prev = self.board[i][j]

                        # shifting right
                        self.board[i][j] = 0
                        self.board[i][idx] = prev
                        idx -= 1
                    else:
                        if prev == self.board[i][j]:
                            self.board[i][idx + 1] += self.board[i][j]
                            merged_tiles_log_sum += (
                                self.board[i][idx + 1].bit_length() - 1
                            )
                            self.max_tile_value = max(
                                self.board[i][idx + 1], self.max_tile_value
                            )
                            self.board[i][j] = 0
                            prev = -1
                        else:
                            prev = self.board[i][j]

                            # shifting right
                            self.board[i][j] = 0
                            self.board[i][idx] = prev
                            idx -= 1
                j -= 1
        return merged_tiles_log_sum

    def action_to_string(self, action: int) -> str:
        return self.actions[action]        

    def play(self, action: int) -> tuple[float, bool]:
        move_fn, is_possible_fn = self.actions_map[action]

        if not is_possible_fn():
            return INVALID_ACTION_PENALTY, False

        total_merged_log_sum = move_fn()
        state_reward = self.state_reward()
        return total_merged_log_sum * self.merge_tile_bonus + state_reward, True

    def state_reward(self) -> float:
        smoothness_penalty = self._smoothness_penalty()
        empty_count = self._count_empty()

        actual_max = self.max_tile_value

        # corner_values = [
        #     self.board[0][0],
        #     self.board[0][3],
        #     self.board[3][0],
        #     self.board[3][3],
        # ]
        # is_max_in_corner = any(v == actual_max for v in corner_values)
        # ! warning
        is_max_in_corner = actual_max == self.board[0][0]
        max_tile_log = actual_max.bit_length() - 1 if actual_max > 0 else 0

        corner_reward = 0
        if is_max_in_corner and actual_max >= 32:
            corner_reward = self.cornered_bonus * max_tile_log

        return (
            (self.smoothness_bonus * smoothness_penalty)
            + (self.empty_count_bonus * empty_count)
            + corner_reward
        )

    def _smoothness_penalty(self) -> int:
        res = 0

        for i in range(4):
            for j in range(4):
                if j < 3:
                    a, b = self.board[i][j], self.board[i][j + 1]
                    if a != 0 and b != 0:
                        res += int(abs(a.bit_length() - b.bit_length()))
                if i < 3:
                    a, b = self.board[i][j], self.board[i + 1][j]
                    if a != 0 and b != 0:
                        res += int(abs(a.bit_length() - b.bit_length()))

        return -res

    def _count_empty(self) -> int:
        empty_count = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_count += 1

        return empty_count

    def play_debug(self) -> None:
        while not self.is_game_over():
            print(self.__str__())
            print(self.get_state_ann())
            action = int(input())
            move, is_possible = self.actions_map[action]
            move()
            self.place_new_block()

        print(self.__str__())

    def _create_new_board(self) -> list[list[int]]:
        return [[0] * 4 for _ in range(4)]

    def sample_actions(self) -> int:
        valid_actions = [i for i, ele in enumerate(self.get_valid_actions_mask()) if ele]
        action = random.choice(valid_actions)
        return action

    def get_state_ann(self) -> list[float]:
        return [(max(x, 1).bit_length() - 1) / 11 for row in self.board for x in row]
        
    def get_state_cnn(self) -> list[list[float]]:
        res = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                res[i][j] = (max(self.board[i][j], 1).bit_length() - 1) / 11

        return res
    
    def get_state(self):
        if (self.learning_mode == "ann"):
            return self.get_state_ann() + self.get_valid_actions_mask()
        elif (self.learning_mode == "cnn"):
            return [self.get_state_cnn() + [self.get_valid_actions_mask()]]
        else:
            raise ValueError("No valid learning mode selected")

    def get_valid_actions_mask(self) -> list[int]:
        return [int(is_possible()) for _, is_possible in self.actions_map]

    def new_game(self) -> None:
        if self.user_board:
            self.board: list[list[int]] = copy.deepcopy(self.user_board)
        else:
            self.board = self._create_new_board()
            self.max_tile_value = self.place_new_block()
            self.max_tile_value = max(self.place_new_block(), self.max_tile_value)

    def __str__(self) -> str:
        res = "-" * 27 + "\n"

        for i in range(4):
            for j in range(4):
                res += f"{self.board[i][j]:4} | " if self.board[i][j] else f"{' ':4} | "

            res += "\n"
            res += "-" * 27 + "\n"

        return res


if __name__ == "__main__":
    board = [[256, 256, 64, 64], [2, 2, 4, 8], [4, 4, 2, 4], [0, 0, 0, 2]]
    game = Game2048(0.9, board=board, learning_mode="ann")
