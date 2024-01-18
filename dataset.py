# dataset.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

import crypto.speck as speck
import crypto.simon as simon

class DatasetGenerator:
    def __init__(self, alg):
        self.alg = alg
    
    def gen_real_dataset(self, input_diff, num_rounds, data_size):
        if self.alg == 'speck_32_64':
            dataset = speck.make_real_data(data_size, num_rounds, input_diff)
        elif self.alg == 'simon_32_64':
            dataset = simon.make_real_data(data_size, num_rounds, input_diff)
        return dataset
    
    def gen_random_dataset(self, num_rounds, data_size):
        if self.alg == 'speck_32_64':
            dataset = speck.make_random_data(data_size, num_rounds)
        elif self.alg == 'simon_32_64':
            dataset = simon.make_random_data(data_size, num_rounds)
        return dataset
    
    def gen_train_dataset(self, input_diff, num_rounds, data_size):
        if self.alg == 'speck_32_64':
            dataset = speck.make_train_data(data_size, num_rounds, input_diff)
        elif self.alg == 'simon_32_64':
            dataset = simon.make_train_data(data_size, num_rounds, input_diff)
        return dataset