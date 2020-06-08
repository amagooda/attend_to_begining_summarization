#!/bin/bash
END=88
for c in $(seq 51 $END); do 
# for c in { 51 .. 88 }; do
    python models/run_sum_clf.py -i forum_coatt_att_combined_aux_ft_gu.yaml --aux_loss --transfer -o ./output/forum_coatt_att_combined/test_${c}/ -j Test -tc forum_coatt_att_combined/forum_coatt_att_combined_aux_ft_gu_${c} -dv cuda:1
done

END=84
for c in $(seq 60 $END); do 
    python models/run_sum_clf.py -i forum_bi_coatt_att_aux_ft_gu.yaml --aux_loss --transfer -o ./output/forum_bi_coatt_att/test_${c}/ -j Test -tc forum_bi_coatt_att/forum_aux_ft_gu_${c} -dv cuda:1
done