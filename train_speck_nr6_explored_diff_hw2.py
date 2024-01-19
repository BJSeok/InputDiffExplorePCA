import train_nets as tn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_diff = (16, 8192)
print(f'HW: 2, speck nr6 - input_diff: {input_diff}')
net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=6, depth=10, alg='speck_32_64', input_diff=input_diff, train_data_size=10**7, eval_data_size=10**6, verbose='1')