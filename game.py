from Exceptions import NoMovesLeft
import random
import numpy as np


class Game2048:
    def __init__(self, prob2: float = 0.9):
        self.prob2 = prob2
        self.prob4 = 1 - prob2
        self.new_game()
        self.actions_map = {
            'U': (self.move_up, self.is_possible_up),
            'D': (self.move_down, self.is_possible_down),
            'L': (self.move_left, self.is_possible_left),
            'R': (self.move_right, self.is_possible_right)
        }

        self.top_row_asc_fill_bonus = 0.2
        self.empty_count_bonus = 0.01
        self.snake_formation_bonus = 0.5

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

    def place_new_block(self) -> None:
        if self.is_game_over():
            raise NoMovesLeft("No moves left")

        empty_slots = self.find_empty_slots()
        i, j = random.choice(empty_slots)
        self.board[i][j] = 2 if random.random() < self.prob2 else 4

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
                            merged_tiles_log_sum += self.board[idx - 1][j].bit_length() - 1
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
                            merged_tiles_log_sum += self.board[idx + 1][j].bit_length() - 1
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
                            merged_tiles_log_sum += self.board[i][idx - 1].bit_length() - 1
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
                            merged_tiles_log_sum += self.board[i][idx + 1].bit_length() - 1
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

    def play(self, action: str) -> float:
        move_fn, is_possible_fn = self.actions_map[action]

        if not is_possible_fn():
            return -1
        
        total_merged_log_sum = move_fn()
        state_reward = self.state_reward()
        self.empty_count_bonus += .005
        print(f"{self.empty_count_bonus = }")
        print(f"state_reward = {state_reward}, {total_merged_log_sum = } \ttotal = {total_merged_log_sum + state_reward}")
        return total_merged_log_sum + state_reward

    def state_reward(self) -> float:
        res = 0

        # more numer of empty boxes is better
        empty_count = 0
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_count += 1

        # rows are -
        #   [ascending ]
        #   [descending]
        #   [ascending ]
        #   [descending]

        snake_formation_count = 0
        is_top_row_ascending = False

        is_going_right = True
        for i in range(4):
            for j in range(3):
                if self.board[i][j] == 0 or self.board[i][j + 1] == 0:
                    continue

                if is_going_right and self.board[i][j] > self.board[i][j + 1]:
                    snake_formation_count += 1
                elif not is_going_right and self.board[i][j] < self.board[i][j + 1]:
                    snake_formation_count += 1

            is_going_right = not is_going_right
            if i == 0 and snake_formation_count == 3:
                is_top_row_ascending = True

        # good if top row is in ascending order
        top_row_asc_fill_bonus = self.top_row_asc_fill_bonus if (0 not in self.board[0]) and is_top_row_ascending else 0

        res += snake_formation_count * (top_row_asc_fill_bonus + self.snake_formation_bonus)
        res += empty_count * self.empty_count_bonus
        print(f"{snake_formation_count = }, {empty_count = }, {top_row_asc_fill_bonus + self.snake_formation_bonus = }")
        return res

    def play_debug(self) -> None:
        while not self.is_game_over():
            print(self.__str__())
            if self.is_possible_left():
                self.play('L')
            elif self.is_possible_up():
                self.play('U')
            elif self.is_possible_right():
                self.play('R')
            elif self.is_possible_down():
                self.play('D')

            print(self.__str__())
            self.place_new_block()
            input()
        print(self.__str__())

    def _create_new_board(self) -> list[list[int]]:
        return [[0] * 4 for _ in range(4)]

    def get_state(self) -> list[int]:
        return [x for row in self.board for x in row]

    def new_game(self) -> None:
        self.board = self._create_new_board()
        self.place_new_block()
        self.place_new_block()

    def __str__(self) -> str:
        res = "-" * 27 + "\n"

        for i in range(4):
            for j in range(4):
                res += f"{self.board[i][j]:4} | " if self.board[i][j] else f"{' ':4} | "

            res += "\n"
            res += "-" * 27 + "\n"

        return res
