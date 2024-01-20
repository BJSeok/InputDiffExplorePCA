import train_nets as tn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for i in range(4):
#     input_diff = (0, 1 << i)
#     print(f'simon nr8 - input_diff: {input_diff}')
#     net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=8, depth=10, alg='simon_32_64', input_diff=input_diff, train_data_size=10**7, eval_data_size=10**6, verbose='0')

input_diff = (0, 0x0004)
print(f'simon nr8 - input_diff: {input_diff}')
net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=8, depth=10, alg='simon_32_64', input_diff=input_diff, train_data_size=10**7, eval_data_size=10**6, verbose='1')

input_diff = (0, 0x0008)
print(f'simon nr8 - input_diff: {input_diff}')
net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=8, depth=10, alg='simon_32_64', input_diff=input_diff, train_data_size=10**7, eval_data_size=10**6, verbose='1')