#!/bin/sh

PROCSTRING_BASENAME=`basename ${PROCSTRING}`
OUTPUT_FILER=${LOGDIR}/${PROCSTRING_BASENAME}.${HOSTNAME}.$$.output

cd /home/lerrelp/rllab/adversary_scripts

#run a niced  
nice python train_trpo_baseline.py --env $ENV --n_exps 5 --n_itr 500 --layer_size 64 64 --batch_size $batch_size --step_size $step_size --gae_lambda $gae_lambda > $OUTPUT_FILER
echo "Finished Without Problems" >> $OUTPUT_FILER
echo "..::RL Solved::.."
