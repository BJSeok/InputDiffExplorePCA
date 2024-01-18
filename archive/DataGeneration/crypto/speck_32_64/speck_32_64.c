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

#include "speck_32_64.h"


const char *cipher_name = CIPHER_NAME;
const int block_size = BLOCK_SIZE;
const int key_size = KEY_SIZE;
const int word_size = WORD_SIZE;

/*
    Key scheduling for encryption
     - mk: master key
     - rk: round keys
*/
int KeySchedule_Encryption(unsigned char *mk, unsigned char *rk, int nr)
{
    // printf("SPECK-32/64 enc_key schedule\n");
    unsigned short *key16 = (unsigned short *)mk;
    unsigned short *roundkeys16 = (unsigned short *)rk;

    unsigned short y = key16[0];
    unsigned short x = key16[1];
    unsigned short key2 = key16[2];
    unsigned short key3 = key16[3];
    unsigned short tmp;

    // printf("key16[0] = %04x\n", key16[0]);
    // printf("key16[1] = %04x\n", key16[1]);
    // printf("key16[2] = %04x\n", key16[2]);
    // printf("key16[3] = %04x\n", key16[3]);

    int i = 0;

    while (1)
    {
        roundkeys16[i] = y;
        // printf("%04x, ", roundkeys16[i]);

        if (i == nr - 1)
            break;

        x = (ROR(x, ALPHA) + y) ^ i++;
        y = ROL(y, BETA) ^ x;

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
 */
int KeySchedule_Decryption(unsigned char *mk, unsigned char *rk, int nr)
{
    // printf("SPECK-32/64 dec_key schedule\n");
    /* Add here the decryption key schedule implementation */
    return 0;
}


/*
    Encryption
    - mesg: plaintext; input messages
    - rk: round keys for encryption
    - nr: number of rounds
*/
int Encryption(unsigned char *mesg, unsigned char *rk, int nr)
{
    // printf("SPECK-32/64 encryption\n");

    unsigned short *block16 = (unsigned short *)mesg;
    const unsigned short *roundkeys = (unsigned short *)rk;

    unsigned short y = block16[0];
    unsigned short x = block16[1];

    // printf("block16[0] = %04x\n", block16[0]);
    // printf("block16[1] = %04x\n", block16[1]);
    unsigned short tmp = ROR(block16[1], ALPHA);
    // printf("ROR(x, ALPHA) = %04x\n", tmp);

    unsigned char i;

    // printf("-----------------------\n");
    for (i = 0; i < nr; ++i)
    {

        x = (ROR(x, ALPHA) + y) ^ roundkeys[i];
        y = ROL(y, BETA) ^ x;
       // printf("%04x %04x\n", y, x);
    }
    // printf("-----------------------\n");

    
    block16[0] = y;
    block16[1] = x;
    

    // printf("ct = %04x%04x\n", x, y);
    

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
    // printf("SPECK-32/64 encryption\n");

    unsigned short *block16 = (unsigned short *)mesg;
    const unsigned short *roundkeys = (unsigned short *)rk;

    unsigned short y = block16[0];
    unsigned short x = block16[1];

    unsigned char i;

    for (i = nr - 1; i >= 0; --i)
    {
        y = ROR(x ^ y, BETA);
        x = ROL((x ^ roundkeys[i]) - y, ALPHA);
    }

    block16[0] = y;
    block16[1] = x;

    return 0;
}