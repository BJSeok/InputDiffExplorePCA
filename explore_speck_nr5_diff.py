import explore_input_diff
import time

print('Explore SPECK-32/64 nr5 input difference...')

print('- Hamming weight: 1')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=1, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_1.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 2')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=2, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_2.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 3')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=3, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_3.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 4')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=4, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_4.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 5')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=5, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_5.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 6')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=6, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_6.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 7')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=7, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_7.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 8')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=8, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_8.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 9')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=9, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_9.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 10')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=10, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_10.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 11')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=11, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_11.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')


print('- Hamming weight: 12')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=12, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_12.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 13')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=13, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_13.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 14')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=14, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_14.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 15')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=15, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_15.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 16')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=16, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_16.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 17')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=17, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_17.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 18')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=18, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_18.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 19')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=19, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_19.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 20')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=20, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_20.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 21')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=21, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_21.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 22')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=22, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_22.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 23')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=23, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_23.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 24')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=24, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_24.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 25')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=25, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_25.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 26')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=26, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_26.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 27')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=27, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_27.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 28')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=28, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_28.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 29')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=29, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_29.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 30')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=30, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_30.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 31')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=31, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_31.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')

print('- Hamming weight: 32')
start_time = time.time()
explore_input_diff.explore_input_difference(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=32, t0=0.003, num_PCs=3, savepath="./explore_results/speck_32_64_nr5_hw_32.txt")
end_time = time.time()
print(f'  * Elapsed time: {end_time - start_time} sec')
