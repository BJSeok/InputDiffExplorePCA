all:blockspair_random blockspair_ciphertext

blockspair_random: blockspair_random.o
	gcc -o blockspair_random blockspair_random.o
blockspair_random.o: blockspair_random.c
	gcc -c -o blockspair_random.o blockspair_random.c

blockspair_ciphertext: blockspair_ciphertext.o
	gcc -o blockspair_ciphertext blockspair_ciphertext.o
blockspair_ciphertext.o: blockspair_ciphertext.c
	gcc -c -o blockspair_ciphertext.o blockspair_ciphertext.c

clean:
	rm -rf *.o
	rm -rf ./blockspair_random
	rm -rf ./blockspair_ciphertext