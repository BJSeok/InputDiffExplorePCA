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
    Block cipher: speck_64_128
    This source was modified by Byoungjin Seok.
    The original source code is Speck_64_128_v01 in FELICS
    About the information of original source as follows:
 */

 /*
 *
 * University of Luxembourg
 * Laboratory of Algorithmics, Cryptology and Security (LACS)
 *
 * FELICS - Fair Evaluation of Lightweight Cryptographic Systems
 *
 * Copyright (C) 2015 University of Luxembourg
 *
 * Written in 2015 by Yann Le Corre <yann.lecorre@uni.lu>,
 *                    Jason Smith <jksmit3@tycho.ncsc.mil>
 *
 * This file is part of FELICS.
 *
 * FELICS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * FELICS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "speck_64_128.h"

const char *cipher_name = CIPHER_NAME;
const int block_size = BLOCK_SIZE;
const int key_size = KEY_SIZE;
const int word_size = WORD_SIZE;

/*
    Key scheduling for encryption
     - mk: master key
     - rk: round keys
     - nr: number of rounds
*/
int KeySchedule_Encryption(unsigned char *mk, unsigned char *rk, int nr)
{
    unsigned int *key32 = (unsigned int *)mk;
    unsigned int *roundKeys32 = (unsigned int *)rk;

    unsigned int y = key32[0];
    unsigned int x = key32[1];
    unsigned int key2 = key32[2];
    unsigned int key3 = key32[3];
    unsigned int tmp;

    unsigned int i = 0;

    while (1)
    {

        roundKeys32[i] = y;

        if (i == NUMBER_OF_ROUNDS - 1)
            break;

        x = (rot32r8(x) + y) ^ i++;
        y = rot32l3(y) ^ x;

        tmp = x;
        x = key2;
        key2 = key3;
        key3 = tmp;
    }

    return 0;
}

/*
    Key scheduling for decryption
     - mk: master key
     - rk: round keys
     - nr: number of rounds
 */
int KeySchedule_Decryption(unsigned char *mk, unsigned char *rk, int nr)
{
    /* Add here the decryption key schedule implementation */

    return 0;
}

/*
    Encryption
    - mesg: plaintext
    - rk: round keys for encryption
    - nr: number of rounds
*/
int Encryption(unsigned char *mesg, unsigned char *rk, int nr)
{
    unsigned int *block32 = (unsigned int *)mesg;
    const unsigned int *roundkeys = (unsigned int *)rk;

    unsigned int y = block32[0];
    unsigned int x = block32[1];

    unsigned char i;

    for (i = 0; i < NUMBER_OF_ROUNDS; ++i)
    {

        x = (rot32r8(x) + y) ^ roundkeys[i];
        y = rot32l3(y) ^ x;
    }

    block32[0] = y;
    block32[1] = x;

    return 0;
}

/*
    Decryption
    - mesg: ciphertext
    - rk: round keys for decryption
    - nr: number of rounds
*/
int Decryption(unsigned char *mesg, unsigned char *rk, int nr)
{
    unsigned int *block32 = (unsigned int *)mesg;
    const unsigned int *roundkeys = (unsigned int *)rk;

    unsigned int y = block32[0];
    unsigned int x = block32[1];

    unsigned char i;

    for (i = NUMBER_OF_ROUNDS - 1; i >= 0; --i)
    {
        y = rot32r3(x ^ y);
        x = rot32l8((x ^ roundkeys[i]) - y);
    }

    block32[0] = y;
    block32[1] = x;

    return 0;
}