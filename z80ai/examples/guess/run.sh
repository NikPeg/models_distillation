#!/bin/sh
#./gendata.py -t 'elephant' -d 30 -n 1000 --nonsense | tee -a elephant.txt
cat *.txt | ./balance.py -t 5000 -o --stats | ../../feedme.py
../../buildz80com.py -o GUESS.COM
../../cpm GUESS.COM
