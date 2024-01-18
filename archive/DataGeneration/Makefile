#
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
#
#
# Copyright (C) 2020 MCRC
#
# Written in 2020 by Byoungjin Seok <sbj7534@seoultech.ac.kr>
#
#

CRYPTO_DIR = ./crypto
INCLUDE_DIR = ./include
ALLCIPHERS = $(wildcard $(CRYPTO_DIR)/*)
CRYPTOMAKE = ./../../include/crypto.mk

.PHONY: all
all:
	@echo Make all macro...
	$(foreach dir, $(ALLCIPHERS), make -C $(dir) -f $(CRYPTOMAKE) $@ &&) true
	@echo End!!


clean :
	@echo Begin cleaning all ciphers - $@
	rm -f *.o
	$(foreach dir, $(ALLCIPHERS), make -C $(dir) -f $(CRYPTOMAKE) $@ &&) true
	@echo End cleaning all ciphers - $@