#!/bin/bash


for i in `seq 0 7`
do 
  echo "$i"
  python3 run_imbalanced.py $i 
done

