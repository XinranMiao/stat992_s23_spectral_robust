#!/bin/bash


for i in `seq 1 35`
do 
  echo "$i"
  python3 run_imbalanced.py $i 
done

