//
//  main.c
//  List2Frame
//
//  Created by 석병진 on 2020/07/25.
//  Copyright © 2020 CIS Lab. All rights reserved.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

void DoProgress(char label[], int step, int total);

int main(int argc, const char* argv[]) {
	unsigned long num;

	char buf[100] = {0, };
	FILE* inputFile = NULL;
	size_t input_size = 0;
	FILE* outputFile = NULL;

	int cnt = 0;
	char* p1;
	char* p2;
	char* c1;
	char* c2;
	char* labelBool;

	clock_t start, end;
	double result;

	start = clock();

	inputFile = fopen(argv[1], "r");
	if (inputFile == NULL)	{
		fputs("File error", stderr);
		exit(1);
	}
	outputFile = fopen(argv[2], "w+");

	fseek(inputFile, 0, SEEK_END);
	input_size = ftell(inputFile);
	rewind(inputFile);

	while (1) {
		fgets(buf, 100, inputFile);

		if(feof(inputFile)){
			break;
		}

		// First column: plain 0
		p1 = strtok(buf, ",");
		
		// Second column: plain 1
		p2 = strtok(NULL, ",");

		// Fifth column: Label
		labelBool = strtok(NULL, ",");

		num = strtoul(p1, NULL, 16);
		for (int i = 0; i < 64; i++) {
			if (num & 0x8000000000000000) {
				fprintf(outputFile, "1,");
			}
			else {
				fprintf(outputFile, "0,");
			}
			num <<= 1;
		}

		num = strtoul(p2, NULL, 16);
		for (int i = 0; i < 64; i++) {
			if (num & 0x8000000000000000) {
				fprintf(outputFile, "1");
			}
			else {
				fprintf(outputFile, "0");
			}
			if ((i + 1) % 64 != 0) {
				fprintf(outputFile, ",");
			}
			else {
				fprintf(outputFile, ",1\n");
			}
			num <<= 1;
		}

		// Progress Bar
		cnt++;
		// DoProgress("Progress", cnt, input_size / 47);
	}
	
	end = clock();
	result = (double)(end - start)/CLOCKS_PER_SEC;
	printf("Execution time: %f s\n", result);
	
	fclose(inputFile);
	fclose(outputFile);

	return 0;
}

void DoProgress(char label[], int step, int total){
	const int pwidth = 72;
	int width = pwidth - strlen(label);
	int pos = (step * width) / total;
	int percent = (step * 100) / total;

	printf("%s:", label);
	
	// ProgressBar
	// printf("%s[", label);
	// for (int i = 0; i < pos; i++) {
	// 	  printf("%c", '=');
	// }

	// printf("% *c", width - pos + 1);
	printf(" %3d%%\r", percent);
}

