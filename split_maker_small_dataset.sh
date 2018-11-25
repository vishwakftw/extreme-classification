#!/bin/bash

if [ "$#" != "4" ] ; then
    echo "Arguments to the script: path to train split file, path to test split file, 
          path to full dataset file where all rows are datapoints, dataset name"
    exit 1
fi

train_split_src=$1
test_split_src=$2
full_dataset_src=$3
dataset_name=$(dirname "$full_dataset_src")"/$4"

echo $dataset_name

# training dataset
while read num rest;
    do
        sed -n "$num p" $full_dataset_src >> "$dataset_name""_train.txt";
    done < $train_split_src

# testing dataset
while read num rest;
    do
        sed -n "$num p" $full_dataset_src >> "$dataset_name""_test.txt";
    done < $test_split_src
