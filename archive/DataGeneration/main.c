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

#include <stdio.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>

#include "crypto.h"

extern const char *cipher_name;
extern const int block_size;
extern const int key_size;
extern const int word_size;

int main(int argc, const char *argv[])
{
    unsigned int state = 0;
    const char dir_path[500] = "dataset/";
    const char *file_name_r = "_Random.csv";
    const char *file_name_d = "_Real.csv";
    int condition_index = 0;

    int data_size = strtoul(argv[3], NULL, 10);
    int num_rounds = strtoul(argv[4], NULL, 10);
    char *tmp_path = strcat(dir_path, cipher_name);
    char *file_path = NULL;

    // Check test vector
    unsigned int test_p1 = 0x6574694c;
    unsigned long long test_key = 0x1918111009080100;
    unsigned char *test_rk = (unsigned char *)calloc(num_rounds * word_size, sizeof(unsigned char));

    KeySchedule_Encryption((unsigned char *)&test_key, test_rk, num_rounds);
    Encryption((unsigned char *)&test_p1, test_rk, num_rounds);

    printf("ct = %08x\n", test_p1);
    

    // printf("cipher: %s\n", cipher_name);
    // printf("target file: %s\n", argv[1]);
    // if (!strcmp(argv[2], "-r"))
    // {
    //     printf("Case = Random\n");
    //     file_path = strcat(tmp_path, file_name_r);
    // }
    // else if(!strcmp(argv[2], "-d"))
    // {
    //     printf("Case = Real\n");
    //     file_path = strcat(tmp_path, file_name_d);
    // }
    // else
    // {
    //     printf("Unknown case!\n");
    //     return 0;
    // }
    // printf("data size: %d\n", data_size);
    // printf("num rounds: %d\n", num_rounds);

    // FILE *fp_r = NULL;

    // // char *file_path = strcat(strcat(dir_path, cipher_name), file_name_r);
    // fp_r = fopen(file_path, "w+");
    // if (fp_r == NULL)
    // {
    //     printf("File open error\n");
    //     exit(1);
    // }

    // unsigned char *mesg_1 = (unsigned char *)calloc(block_size, sizeof(unsigned char));
    // unsigned char *mesg_2 = (unsigned char *)calloc(block_size, sizeof(unsigned char));
    // unsigned char *mk_1 = (unsigned char *)calloc(key_size, sizeof(unsigned char));
    // unsigned char *mk_2 = (unsigned char *)calloc(key_size, sizeof(unsigned char));
    // unsigned char *rk_1 = (unsigned char *)calloc(num_rounds * word_size, sizeof(unsigned char));
    // unsigned char *rk_2 = (unsigned char *)calloc(num_rounds * word_size, sizeof(unsigned char));
    // unsigned char label;
    // unsigned char *tmp1, *tmp2, *tmp3, *tmp4;
    // unsigned int plain0, plain1;
    // unsigned long long key;
    // int cnt = 0;
    
    // char buf[1024];
    // FILE *pFile = NULL;
    // pFile = fopen(argv[1], "r");
    // if (pFile != NULL)
    // {
    //     while(!feof(pFile))
    //     {
    //         fgets(buf, 1024, pFile);
            
    //         if(cnt == data_size)
    //         {
    //             break;
    //         }
                
            
    //         tmp1 = strtok(buf, ",");
    //         tmp2 = strtok(NULL, ",");
    //         tmp3 = strtok(NULL, ",");
    //         tmp4 = strtok(NULL, ",");

    //         plain0 = strtoul(tmp1, NULL, 16);
    //         plain1 = strtoul(tmp2, NULL, 16);
    //         key = strtoull(tmp3, NULL, 16);
    //         label = strtoul(tmp4, NULL, 16);
    //         // printf("%08X\n", plain0);
    //         // printf("%08X\n", plain1);
    //         // printf("%016llX\n", key);
    //         // printf("%d\n", label);

    //         // Encryption
    //         // Key schedule & Encryption (1st column)

    //         KeySchedule_Encryption((unsigned char *)&key, rk_1, num_rounds);
    //         Encryption((unsigned char *)&plain0, rk_1, num_rounds);

    //         // Key schedule & Encryption (2nd column)
    //         KeySchedule_Encryption((unsigned char *)&key, rk_2, num_rounds);
    //         Encryption((unsigned char *)&plain1, rk_2, num_rounds);

    //         // printf("ct0 = %08X\n", plain0);
    //         // printf("ct1 = %08X\n", plain1);

    //         // make the result file
    //         fprintf(fp_r, "0x%08X,", plain0);
    //         fprintf(fp_r, "0x%08X,", plain1);
    //         fprintf(fp_r, "%d\n", label);
    //         cnt++;
    //     }
    // }

    // fclose(fp_r);
    // fclose(pFile);
    
    printf("Done!\n");

    return 0;
}