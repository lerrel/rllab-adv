#!/bin/sh

PROCSTRING_BASENAME=`basename ${PROCSTRING}`
OUTPUT_FILER=${LOGDIR}/${PROCSTRING_BASENAME}.${HOSTNAME}.$$.output

cd /home/lerrelp/rllab/adversary_scripts

#run a niced  
nice python train_trpo_var_adversary.py --env $ENV --adv_name no_adv --n_exps 5 --n_itr 500 --layer_size 64 64 --batch_size $batch_size --step_size $step_size --gae_lambda $gae_lambda --adv_fraction $adv_fraction> $OUTPUT_FILER
echo "Finished Without Problems" >> $OUTPUT_FILER
echo "..::RL Solved::.."
