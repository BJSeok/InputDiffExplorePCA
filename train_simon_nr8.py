import train_nets as tn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for i in range(8):
    input_diff = (0, 1 << (i+9))
    print(f'simon nr8 - input_diff: {input_diff}')
    net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=8, depth=10, alg='simon_32_64', input_diff=input_diff, train_data_size=10**7, eval_data_size=10**6, verbose='0')
