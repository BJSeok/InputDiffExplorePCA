import train_nets as tn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

net, h = tn.train_neural_distinguisher(num_epochs=200, num_rounds=9, depth=10, alg='simon_32_64', input_diff=(0x0, 0x0080), train_data_size=10**7, eval_data_size=10**6)
