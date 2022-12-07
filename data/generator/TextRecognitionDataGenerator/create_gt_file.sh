#!/bin/bash

export LANG=ko_KR.utf8

# echo $1

searchdir=./out/$1

for entry in $searchdir/*

do
        filedir="${entry:2}"
        filename="${filedir##*/}"
        label="${filename%_*}"
        echo -e "$filedir\t$label"

done