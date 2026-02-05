#!/bin/sh
zcat training-data.txt.gz | ../../feedme.py 
../../buildz80com.py -o CHAT.COM
../../cpm CHAT.COM