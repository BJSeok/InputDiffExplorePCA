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

#ifndef CRYPTO_H
#define CRYPTO_H

/*
    Key scheduling for encryption
     - mk: master key
     - rk: round keys
     - nr: number of rounds
*/
int KeySchedule_Encryption(unsigned char *mk, unsigned char *rk, int nr);

/*
    Key scheduling for decryption
     - mk: master key
     - rk: round keys
     - nr: number of rounds
 */
int KeySchedule_Decryption(unsigned char *mk, unsigned char *rk, int nr);


/*
    Encryption
    - mesg: plaintext; input messages
    - rk: round keys for encryption
    - nr: number of rounds
*/
int Encryption(unsigned char *mesg, unsigned char *rk, int nr);

/*
    Decryption
    - mesg: ciphertext
    - rk: round keys for decryption
    - nr: number of rounds
*/
int Decryption(unsigned char *mesg, unsigned char *rk, int nr);

#endif /* CRYPTO_H */