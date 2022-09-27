import random
import math

class InvertibleBloomFilterSimulation:
    def __init__(self, num_hash_functions, num_elements_inserted, additional_space):
        self.r, self.n, self.c = num_hash_functions, num_elements_inserted, additional_space
        self.m = self.r * self.n * additional_space

        self.rounds, self.left_over_elements = -1, -1

        self.table = [0] * self.m
        self.place_map = [0] * self.m
        self.element_map = [[] for _ in range(self.n)]

    def simulate_single_insert(self, inserted_element, place):
        # Increment the value at place to know how many elements are inserted there
        self.table[place] += 1

        # Add the place being inserted into a map containing the inserted element
        # The values are added up so if there is a single element inserted, it will be true
        # Otherwise the point is corrupted and we need to remove elements from that place
        self.place_map[place] += inserted_element

        # If the value at place is 1, then we know the value at place_map is true
        # We can use that value to then lead to all other places that the element was inserted
        # From there, we can delete them
        self.element_map[inserted_element].append(place)

    def simulate_insertions(self):
        for i in range(self.n):
            for _ in range(self.r):
                self.simulate_single_insert(i, random.randint(0, self.m - 1))

    def simulate_deletions(self):
        rounds, continue_flag = 0, True
        # Cotinuously loop through the table, looking for points in the table that have a
        # value of 1. Then find the element that was inserted and all other places that it 
        # mapped to and remove them from the table. Continue until there are no more cells
        # in the table with a value of 1
        while continue_flag:
            continue_flag = False
            rounds += 1
            for i in range(self.m):
                if self.table[i] == 1:
                    continue_flag = True
                    element = self.place_map[i]
                    for place in self.element_map[element]:
                        self.table[place] = 0
                        self.place_map[place] = 0
                    self.element_map[element] = []

        self.rounds = rounds - 1

    def check_all_elements_deleted(self):
        self.left_over_elements = 0
        for i in self.table:
            self.left_over_elements += i
            if i < 0:
                self.negative_error()

    def negative_error(self):
        print("ERROR: There was a negative value found within the table!!!\n")
        self.print_debug_info()
        exit(1)

    def run_simulation(self):
        print("RUNNING TEST")
        print("n:", self.n)
        print("r:", self.r)
        print("c':", self.c, "\n")

        self.simulate_insertions()
        self.simulate_deletions()
        self.check_all_elements_deleted()

        if self.left_over_elements == 0:
            print("ALL ELEMENTS DELTED")
        else:
            print(self.left_over_elements, "REMAINS IN THE FILTER")

        print("REMOVAL PROCESS TOOK", self.rounds, "ROUNDS")

    def print_debug_info(self):
        print("Table:")
        print(self.table)
        print("Place Map:")
        print(self.place_map)
        print("Element Map:")
        print(self.element_map)
        print("\n")

n = 1000000
r, c = math.floor((math.log(n))), 2
test = InvertibleBloomFilterSimulation(r, n, c)
test.run_simulation()
