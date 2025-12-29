import copy
import random
import numpy as np

INVANLID_ACTION_PENALTY = -5


class Game2048:
    def __init__(self, prob2: float = 0.9, board=None):
        self.prob2 = prob2
        self.prob4 = 1 - prob2

        self.user_board = None
        if board:
            self.user_board = board.copy()

        self.max_tile_value = 0
        self.new_game()
        self.actions_map = {
            "U": (self.move_up, self.is_possible_up),
            "D": (self.move_down, self.is_possible_down),
            "L": (self.move_left, self.is_possible_left),
            "R": (self.move_right, self.is_possible_right),
        }
        self.actions = list(self.actions_map.keys())

        self.merge_tile_bonus = 1
        self.empty_count_bonus = 0.1
        self.cornered_bonus = 0.5
        self.smoothness_bonus = 0.015
        # self.empty_count_bonus = 0.0
        # self.smoothness_bonus = 0.0
        # self.cornered_bonus = 2

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

    def play(self, action: str) -> tuple[float, bool]:
        move_fn, is_possible_fn = self.actions_map[action]

        if not is_possible_fn():
            return INVANLID_ACTION_PENALTY, False

        total_merged_log_sum = move_fn()
        state_reward = self.state_reward()
        return total_merged_log_sum * self.merge_tile_bonus + state_reward, True

    def state_reward(self) -> float:
        smoothness_penalty = self._smoothness_penalty()
        empty_count = self._count_empty()

        actual_max = self.max_tile_value
        corner_values = [
            self.board[0][0],
            self.board[0][3],
            self.board[3][0],
            self.board[3][3],
        ]

        is_max_in_corner = any(v == actual_max for v in corner_values)
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
                        res += int(abs(np.log2(a) - np.log2(b)))
                if i < 3:
                    a, b = self.board[i][j], self.board[i + 1][j]
                    if a != 0 and b != 0:
                        res += int(abs(np.log2(a) - np.log2(b)))

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
            action = input()
            move, is_possible = self.actions_map[action]
            move()
            self.place_new_block()

        print(self.__str__())

    def _create_new_board(self) -> list[list[int]]:
        return [[0] * 4 for _ in range(4)]

    def sample_actions(self) -> str:
        action = random.choice(self.actions)
        return action

    def get_state_ann(self) -> list[float]:
        return [
            (max(x, 1).bit_length() - 1) / 11 for row in self.board for x in row
        ] + self.get_valid_actions_mask()

    def get_state_cnn(self):
        res = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                res[i][j] = (max(self.board[i][j], 1).bit_length() - 1) / 11

        return [res]

    def get_valid_actions_mask(self) -> list[int]:
        return [int(is_possible()) for _, is_possible in self.actions_map.values()]

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
    game = Game2048(0.9, board)
