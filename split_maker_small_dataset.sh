#!/bin/bash

train_split_src="./data/Bibtex/bibtex_trSplit.txt"
test_split_src="./data/Bibtex/bibtex_tstSplit.txt"
full_dataset_src="./data/Bibtex/Bibtex_data.txt"
dataset_name="./data/Bibtex/bibtex"

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
