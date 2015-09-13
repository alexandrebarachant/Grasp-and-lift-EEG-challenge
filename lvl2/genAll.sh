#!/usr/bin/env sh
array=( val test )

for i in "${array[@]}"
do
  for filename in models/*.yml; do
    echo "$filename"
    if [[ "$filename" == *"bags_model"* ]]
    then
      python genEns_BagsModels.py $filename $i     
    elif [[ "$filename" == *"bags"* ]]
    then
      python genEns_BagsSubjects.py $filename $i
    else
      python genEns.py $filename $i
    fi
  done
done