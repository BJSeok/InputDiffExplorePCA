MAIN_DIR = ./../../
INCLUDE_DIR = ./../../include

CURRENTPATHDIRS = $(subst /, , $(CURDIR))
CIPHERNAME = $(word $(words $(CURRENTPATHDIRS)), $(CURRENTPATHDIRS))

SOURCES = $(wildcard $(CURDIR)/*.c)
OBJS = $(subst $(CURDIR)/, , $(SOURCES:.c=.o))

INCLUDES = -I$(INCLUDE_DIR)

.PHONY: all

all : exec-build

.PHONY : exec-build
exec-build : main-build \
		Encryptor_$(CIPHERNAME).elf
	@echo -------- End building $(CIPHERNAME) --------

.PHONY : main-build
main-build : \
		cipher-build \
		main.o

.PHONY : cipher-build
cipher-build : $(OBJS)
	@echo -------- Start building $(CIPHERNAME) --------

%.o : \
		%.c \
		$(INCLUDE_DIR)/crypto.h
	$(CC) -c $(CFLAGS) $< $(INCLUDES) -o $@

main.o : \
		$(MAIN_DIR)/main.c \
		$(INCLUDE_DIR)/crypto.h
	$(CC) -c $(CFLAGS) $< $(INCLUDES) -o $(MAIN_DIR)/$@

Encryptor_$(CIPHERNAME).elf : $(OBJS) main.o
	$(CC) -o $(MAIN_DIR)/Encryptor_$(CIPHERNAME).elf $(OBJS) $(MAIN_DIR)/main.o

clean:
	@echo Begin cleaning: $(CIPHERNAME)
	rm -f *.o
	rm -f $(MAIN_DIR)/*.o
	rm -f $(MAIN_DIR)/*.elf
	@echo End cleaning: $(CIPHERNAME)


