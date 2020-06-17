#!/bin/bash
END=88
for c in $(seq 0 $END); do
    ### Run on validatin set
    python models/run_sum_clf.py -i forum_coatt_att_combined_aux_ft_gu.yaml --aux_loss --transfer -o ./output/forum_coatt_att_combined/val_${c}/ -j Val -tc forum_coatt_att_combined/forum_coatt_att_combined_aux_ft_gu_${c} -dv cuda:1
    ### Run on Test set
    python models/run_sum_clf.py -i forum_coatt_att_combined_aux_ft_gu.yaml --aux_loss --transfer -o ./output/forum_coatt_att_combined/test_${c}/ -j Test -tc forum_coatt_att_combined/forum_coatt_att_combined_aux_ft_gu_${c} -dv cuda:1
done