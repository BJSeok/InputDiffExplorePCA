# datagen_helper.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

import os
import subprocess

def gen_data(algo, n, nr, diff):
    """Generate block cipher algorithm dataset and return dataframe filepath

    Parameters
    ----------
    algo : string
        Algorithm name
    n : string
        Data size
    nr : string
        Round number
    diff : string
        Differencial

    Returns
    -------
    df_pair_random_filepath : string
    df_pair_cipher_filepath : string
        Dataframe filepaths of block cipher algorithm dataset

    """
    os.chdir("./DataGeneration")
    subprocess.check_output("make")

    df_pair_random_filepath = "./DataGeneration/dataset/df_" + algo + "_RandomNumber.csv"
    df_pair_cipher_filepath = "./DataGeneration/dataset/df_" + algo + "_Differential.csv"

    subprocess.check_output("./Encryptor_" + algo + ".elf -r --condition " + n + " " + nr, shell=True)
    subprocess.check_output("./Encryptor_" + algo + ".elf -d --condition " + n + " " + nr + " " + diff, shell=True)

    os.chdir("../")

    subprocess.check_output("./list2frame_" + algo.split("_")[1] +  "/blockspair_random ./DataGeneration/dataset/" + algo + "_RandomNumber.csv " + df_pair_random_filepath, shell=True)
    subprocess.check_output("./list2frame_" + algo.split("_")[1] +  "/blockspair_ciphertext ./DataGeneration/dataset/" + algo + "_Differential.csv " + df_pair_cipher_filepath, shell=True)

    return df_pair_random_filepath, df_pair_cipher_filepath