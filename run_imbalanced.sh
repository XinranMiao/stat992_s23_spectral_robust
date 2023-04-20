#!/bin/bash


for i in `seq 4 10`
do 
  echo "$i"
  python3 run_imbalanced.py $i 
done

