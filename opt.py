import numpy as np
import time
import random
import matplotlib.pyplot as plt
import json
import math


def time_of_work(func):
    import time

    def wrapper(self, *args, **kwargs):
        start = time.time()
        func(self, *args, **kwargs)
        end = time.time()
        print("[Function time]:", func.__name__, "[time]:", end - start)

    return wrapper


class DataMatrix:

    def __init__(self):
        self.__data_file_csv = "Data/Circle_coord.csv"
        self.__data = None
        self.count_of_data = None
        self.__distance_matrix = None
        self.available_data_distance_matrix = None
        self.available_data = None
        self.distance_matrix_shape = None

    def read_file(self):
        self.__data = np.genfromtxt(self.__data_file_csv, delimiter=",")

    def print_data(self):
        print(self.__data)

    def __preprocess_data(self):
        self.__data = np.delete(self.__data, 0, axis=0)
        self.available_data = self.__data

    def init_distance_matrix(self):
        self.read_file()
        self.__preprocess_data()

        self.count_of_data = int(self.__data.shape[0])

        x = np.array(self.__data[:, 1])
        y = np.array(self.__data[:, 2])
        dx = x[..., np.newaxis] - x[np.newaxis, ...]
        dy = y[..., np.newaxis] - y[np.newaxis, ...]
        d = np.array([dx, dy])
        self.__distance_matrix = (d ** 2).sum(axis=0) ** 0.5
        self.distance_matrix_shape = self.__distance_matrix.shape
        self.available_data_distance_matrix = self.__distance_matrix

    def get_inverse_distance(self, first_point, second_point):
        if first_point == second_point:
            return 0
        else:
            return 1 / self.__distance_matrix[first_point, second_point]

    def print_distance_matrix(self):
        print(self.__distance_matrix)

    def get_distance(self, first_point, second_point):
        return self.__distance_matrix[first_point, second_point]

    def get_path_length(self, path_list):
        p_length = 0
        for i in range(len(path_list) - 1):
            p_length += self.get_distance(path_list[i], path_list[i + 1])
        p_length += self.get_distance(path_list[0], path_list[len(path_list) - 1])
        return p_length

    def set_reading_file(self, file_name):
        self.__data_file_csv = file_name


class ACO:

    def __init__(self):
        self.data_matrices = DataMatrix()
        # =========================================[MATRICES]===========================================================
        # self.__data_file_csv = "DATA_SOURCE.csv"
        # self.__data = None
        # self.__distance_matrix = None
        self.__pheromone_matrix = None
        self.__ant_path_matrix = None
        self.__probability_matrix = None
        self.__best_order = None
        self.__best_path_length = np.inf
        # =========================================[ACO_SETTINGS]=======================================================
        # Count of ants in each generation
        self.ant_count = 30
        # Count of generation(iteration)
        self.generations_count = 50
        # Alpha
        self.alfa = 3
        # Beta
        self.beta = 2
        # Initial pheromone [tau]
        self.initial_pheromone = 0.1
        # Pheromone quantity
        self.pheromone_quantity = 1
        # Pheromone evaporation coefficient [p]
        self.evaporation_coefficient = 0.1
        # =========================================[NUMPY_SETTINGS]=====================================================
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(threshold=np.inf)

    def __str__(self):
        return "[ant_count]: {}\n[generations_count]: {}\n[alfa]: {}\n[beta]: {}\n[initial_pheromone]: {}\n[" \
               "pheromone_quantity]: {}\n[evaporation_coefficient]: {}\n".format(self.ant_count,
                                                                                 self.generations_count, self.alfa,
                                                                                 self.beta, self.initial_pheromone,
                                                                                 self.pheromone_quantity,
                                                                                 self.evaporation_coefficient)

    def read_parameters_file(self, parameters_file_json):
        with open(parameters_file_json, "r") as read_file:
            data = json.load(read_file)
        self.ant_count = data["ACO_parameters"][0]["ant_count"]
        self.generations_count = data["ACO_parameters"][0]["generations_count"]
        self.alfa = data["ACO_parameters"][0]["alfa"]
        self.beta = data["ACO_parameters"][0]["beta"]
        self.initial_pheromone = data["ACO_parameters"][0]["initial_pheromone"]
        self.pheromone_quantity = data["ACO_parameters"][0]["pheromone_quantity"]
        self.evaporation_coefficient = data["ACO_parameters"][0]["evaporation_coefficient"]
        self.data_matrices.set_reading_file(data["ACO_parameters"][0]["__data_file_csv"])
        # self.__data_file_csv = data["ACO_parameters"][0]["__data_file_csv"]
        read_file.close()

    @time_of_work
    def __init_pheromone_matrix(self):
        self.__pheromone_matrix = np.full(self.data_matrices.distance_matrix_shape, self.initial_pheromone, dtype=float)

    def get_pheromone_value(self, current_position, perspective_position):
        return self.__pheromone_matrix[current_position, perspective_position]

    def __init_probability_matrix(self):
        self.__probability_matrix = self.__pheromone_matrix * np.divide(1,
                                                                        self.data_matrices.available_data_distance_matrix,
                                                                        out=np.zeros_like(
                                                                            self.data_matrices.available_data_distance_matrix),
                                                                        where=self.data_matrices.available_data_distance_matrix != 0)

    def __init_ant_path_matrix(self):
        self.__ant_path_matrix = np.zeros(shape=(self.ant_count, self.data_matrices.count_of_data))

    def __get_new_point(self, current_point):
        a = self.__probability_matrix[current_point]
        # print(a)
        next_point = (a[:] / a.sum()).argmax()
        return next_point

    def __set_to_tabu(self, tabu_point):
        self.__probability_matrix[tabu_point, :] = 0
        self.__probability_matrix[:, tabu_point] = 0

    def __set_point_to_path(self, ant_number, point, iteration):
        self.__ant_path_matrix[ant_number, iteration + 1] = point

    def __set_random_point(self, ant_number):
        self.__ant_path_matrix[ant_number, 0] = random.randint(0, self.data_matrices.count_of_data - 1)

    def __fill_ant_path_matrix(self):
        for ant in range(self.ant_count):
            self.__set_random_point(ant)
            self.__init_probability_matrix()
            # print(self.__probability_matrix)
            # print("===========================================================")
            for iteration in range(0, self.data_matrices.count_of_data - 1):
                # print("_______________________________________________")
                # print("[Iteration]:", iteration)
                current_point = int(self.__ant_path_matrix[ant, iteration])
                # print("[current_point]:", current_point)
                next_point = self.__get_new_point(current_point)
                # print("[next_point]:", next_point)
                self.__set_to_tabu(current_point)
                self.__set_point_to_path(ant, next_point, iteration)

    def __get_path_length(self, ant_number):
        p_length = 0
        for i in range(int(self.__ant_path_matrix.shape[1]) - 1):
            p_length += self.data_matrices.get_distance(int(self.__ant_path_matrix[ant_number, i]),
                                                        int(self.__ant_path_matrix[ant_number, i + 1]))

        p_length += self.data_matrices.get_distance(int(self.__ant_path_matrix[ant_number, 0]),
                                                    int(self.__ant_path_matrix[ant_number,
                                                                               int(self.__ant_path_matrix.shape[
                                                                                       1]) - 1]))

        return p_length

    def __get_best(self):
        for ant in range(self.ant_count):
            p_length = self.__get_path_length(ant)
            # print("[Ant]:", ant, "[length]:", p_length)
            if p_length < self.__best_path_length:
                self.__best_path_length = p_length
                self.__best_order = self.__ant_path_matrix[ant]

    def __run_one_generation(self):
        self.__init_ant_path_matrix()
        self.__fill_ant_path_matrix()

        # print(self.__ant_path_matrix)
        self.__get_best()

    def __set_pheromone(self):
        for ant in range(self.ant_count):
            pheromone = self.pheromone_quantity / self.__get_path_length(ant)
            for i in range(0, self.__ant_path_matrix.shape[1] - 1):
                self.__pheromone_matrix[
                    int(self.__ant_path_matrix[ant, i]), int(self.__ant_path_matrix[ant, i + 1])] += pheromone
                self.__pheromone_matrix[
                    int(self.__ant_path_matrix[ant, i + 1]), int(self.__ant_path_matrix[ant, i])] += pheromone

    def __refresh_pheromone(self):
        self.__pheromone_matrix = self.__pheromone_matrix * (1 - self.evaporation_coefficient)

    def plot_path(self):
        x = []
        y = []
        for i in range(len(self.__best_order)):
            x.append(self.data_matrices.available_data[int(self.__best_order[i]), 1])
            y.append(self.data_matrices.available_data[int(self.__best_order[i]), 2])
            plt.plot(self.data_matrices.available_data[int(self.__best_order[i]), 1],
                     self.data_matrices.available_data[int(self.__best_order[i]), 2], marker='D')
            plt.annotate(str(int(self.__best_order[i])), xy=(x[i], y[i]), xytext=(x[i] + 0.5, y[i] + 0.5), size=8)
        plt.plot(x, y, linewidth=0.5)
        # plt.show(block=False)
        plt.savefig('foo.png', dpi=1000)
        plt.show()
        # time.sleep(2)
        # plt.cla()

    # =========================================[MAIN]===================================================================
    def solve(self):
        st = time.time()

        self.data_matrices.init_distance_matrix()
        self.__init_pheromone_matrix()

        for gen in range(self.generations_count):
            self.__run_one_generation()
            self.__set_pheromone()
            self.__refresh_pheromone()
        end = time.time()

        print("__________________________________________________________________________________________")
        print("[Total time]: ", end - st)
        print("[Best path length]:", self.__best_path_length)
        print("[Best order]:", self.__best_order)


class SA:

    def __init__(self):
        self.data_matrices = DataMatrix()
        # ==========================================[SA_SETTINGS]=======================================================
        self.initial_temperature = 10
        self.final_temperature = 0.00001
        self.limit_of_iterations = 500000
        self.best_sequence = None
        self.current_sequence = None
        self.path_length = None

        # =========================================[NUMPY_SETTINGS]=====================================================
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(threshold=np.inf)

    def read_parameters_file(self, parameters_file_json):
        with open(parameters_file_json, "r") as read_file:
            data = json.load(read_file)
        self.initial_temperature = data["SA_parameters"][0]["initial_temperature"]
        self.final_temperature = data["SA_parameters"][0]["final_temperature"]
        self.limit_of_iterations = data["SA_parameters"][0]["limit_of_iterations"]
        # data_matrices.__data_file_csv = data["SA_parameters"][0]["__data_file_csv"]
        self.data_matrices.set_reading_file(data["SA_parameters"][0]["__data_file_csv"])
        read_file.close()

    @staticmethod
    def get_probability_of_permutation(delta_energy, temperature):
        if delta_energy <= 0:
            return 1
        else:
            probability = math.exp(-delta_energy / temperature)
            return probability

    def __generate_initial_sequence(self):
        self.current_sequence = np.random.permutation(self.data_matrices.count_of_data)

    def __random_permutation(self):
        for _ in range(5):
            i = random.randint(0, self.data_matrices.count_of_data - 1)
            j = random.randint(0, self.data_matrices.count_of_data - 1)
            tem = self.current_sequence[i].copy()
            self.current_sequence[i] = self.current_sequence[j].copy()
            self.current_sequence[j] = tem
        """
        for _ in range(5):
            i = random.randint(0, self.data_matrices.count_of_data - 1)
            j = random.randint(0, self.data_matrices.count_of_data - 1)
            if i > j:
                self.current_sequence[j:i] = np.flipud(self.current_sequence[j:i])
            else:
                self.current_sequence[i:j] = np.flipud(self.current_sequence[i:j])
        """

    def decrease_temperature(self, iteration):
        if iteration != 0:
            return self.initial_temperature * 0.1 / iteration
            # Больцмановский отжиг
            # return self.initial_temperature / (math.log(1 + iteration))
            # Быстрый отжиг/ Отжиг Коши
            # return self.initial_temperature / iteration ** (1 / len(self.current_sequence))
            #
            # return self.initial_temperature * 0.5
        else:
            pass

    @staticmethod
    def __is_transition(probability):
        rand_value = random.random()
        # rand_value = 0.3
        if rand_value <= probability:
            return True
        else:
            return False

    def plot(self):
        x = []
        y = []
        for i in range(len(self.best_sequence)):
            x.append(self.data_matrices.available_data[int(self.best_sequence[i]), 1])
            y.append(self.data_matrices.available_data[int(self.best_sequence[i]), 2])
            plt.plot(self.data_matrices.available_data[int(self.best_sequence[i]), 1],
                     self.data_matrices.available_data[int(self.best_sequence[i]), 2], marker='D')
            plt.annotate(str(int(self.best_sequence[i])), xy=(x[i], y[i]), xytext=(x[i] + 0.5, y[i] + 0.5), size=8)
        plt.plot(x, y, linewidth=0.5)
        # plt.show(block=False)
        plt.savefig('foo.png', dpi=1000)
        plt.show()

    def solve(self):
        self.data_matrices.init_distance_matrix()
        self.__generate_initial_sequence()
        self.best_sequence = self.current_sequence.copy()

        iteration = 1

        start = time.time()
        current_temp = self.initial_temperature
        
        while current_temp > self.final_temperature:
            
            self.__random_permutation()
            
            delta_energy = self.data_matrices.get_path_length(self.current_sequence) - \
                           self.data_matrices.get_path_length(self.best_sequence)
            if delta_energy <= 0:
                self.best_sequence = self.current_sequence.copy()
            else:
                if self.__is_transition(self.get_probability_of_permutation(delta_energy, current_temp)):
                    self.best_sequence = self.current_sequence.copy()

            current_temp = self.decrease_temperature(iteration)

            iteration += 1
            if iteration > self.limit_of_iterations:
                break

        end = time.time()

        print("__________________________________________________________________________________________")
        print("[Count of iterations]:", iteration)
        print("[Total time]: ", end - start)
        print("[Best path length]:", self.data_matrices.get_path_length(self.best_sequence))
        print("[Best order]:", self.best_sequence)

