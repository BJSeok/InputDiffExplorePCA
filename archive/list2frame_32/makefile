all: l2f_random l2f_real

l2f_random: l2f_random.o
	gcc -o l2f_random l2f_random.o
l2f_random.o: l2f_random.c
	gcc -c -o l2f_random.o l2f_random.c

l2f_real: l2f_real.o
	gcc -o l2f_real l2f_real.o
l2f_real.o: l2f_real.c
	gcc -c -o l2f_real.o l2f_real.c

clean:
	rm -rf *.o
	rm -rf ./l2f_random
	rm -rf ./l2f_real