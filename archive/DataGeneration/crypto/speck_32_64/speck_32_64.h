/*
 *
 * Seoul National University of Science and Technology (SeoulTech)
 * Cryptography and Information Security Lab.
 *
 *
 * Copyright (C) 2020 MCRC
 *
 * Written in 2020 by Byoungjin Seok <sbj7534@seoultech.ac.kr>
 *
 */

 /*
    Block cipher: speck_32_64
 */

#include <stdio.h>
#include "crypto.h"

// Essential parameters (BYTE)
#define CIPHER_NAME "speck_32_64"
#define BLOCK_SIZE 4
#define KEY_SIZE 8
#define WORD_SIZE 2


#define WORD_SIZE_BIT 16
#define ALPHA 7
#define BETA 2

#define ROL(X, K) ((X << K) | (X >> (WORD_SIZE_BIT - K)))
#define ROR(X, K) ((X >> K) | (X << (WORD_SIZE_BIT - K)))

