#!/usr/bin/env sh
array=( val test )

filenames=( xgb_bags xgb_bags_delay xgb_bags_model xgb_short RNN_256_delay4000_allModels_ADAM_bags RNN_256_delay4000_allModels_ADAM_bags_model RNN_256PR_delay4000_allModels_ADAM_bags_model RNN_256_delay2000_allModels_ADAM_bags_model )


for i in "${array[@]}"
do
  for f in "${filenames[@]}"; do
    filename="models/$f.yml"
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