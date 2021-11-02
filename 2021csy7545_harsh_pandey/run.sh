#!/bin/bash

echo 'Running Code'
echo $1

if [ $1 == '1' ]
then
	python Q1.py $2 $3 $4
elif [ $1 == '2' ]; then
	python Q2.py $2 $3 $4 $5
else
	echo 'Invalid ip'
fi

