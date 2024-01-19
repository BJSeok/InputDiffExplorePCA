import dataset
import pca_helper
import clustering_helper
import random
import math
import time

def generate_numbers_with_hamming_weight(bit_size = 32, hamming_weight = 1, number_pool=None):
    number = 0
    while number == 0:
        bit_position = random.sample(range(bit_size), hamming_weight)
        for position in bit_position:
            number |= (1 << position)
        if number_pool is not None:
            if number in number_pool:
                number = 0
            else:
                number_pool.append(number)
                return number
        else:
            return number

def calculate_combinations(m, n):
    return math.factorial(m) // (math.factorial(n) * math.factorial(m - n))

def explore_input_difference(alg='speck_32_64', blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=1, t0=0.003, num_PCs=3, savepath=None):
    numbers = []
    lambda_base = 1/(2*blocksize)
    dataset_generator = dataset.DatasetGenerator(alg=alg)
    num_cases = calculate_combinations(blocksize, hamming_weight)
    while len(numbers) < num_cases:
        number = generate_numbers_with_hamming_weight(blocksize, hamming_weight, numbers)
        left_word = (number >> wordsize) & 0xFFFF
        right_word = number & 0xFFFF
        diff = (left_word, right_word)
        
        data_speck = dataset_generator.gen_real_dataset(diff, nr, datasize)
        eigen_value, eigen_vector = pca_helper.EigenValueDecomposition(dataset=data_speck)
        if sum(eigen_value - lambda_base > t0) >= num_PCs:
            pca_results = pca_helper.DimensionReduction(data_speck, n_components=3)
            start_time = time.time()
            labels = clustering_helper.kmeans_clustering(pca_results, 27, 3)
            score = clustering_helper.calculate_silhouette(pca_results, labels)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if savepath is not None:
                with open(savepath, "a") as file:
                    file.write(f'diff = {diff}, silhouette score: {score}, elapsed time: {elapsed_time} sec\n')
            else:
                print(f'diff = {diff}, silhouette score: {score}, elapsed time: {elapsed_time} sec')