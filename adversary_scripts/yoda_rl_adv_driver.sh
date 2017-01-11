#!/bin/sh

if [ ! -n "$env" ]
    then
    echo "env UNDEFINED, not running anything"
    exit
fi
if [ ! -n "$adv_fraction" ]
    then
    echo "adv_fraction UNDEFINED, not running anything"
    exit
fi
if [ ! -n "$PROCSTRING" ]
    then
    echo "PROCSTRING UNDEFINED, not running anything"
    exit
fi

PROCSTRING_BASENAME=`basename ${PROCSTRING}`
OUTPUT_FILER=/home/${USER}/tmpoutputs/${PROCSTRING_BASENAME}.${HOSTNAME}.$$.output

cd /home/lerrelp/rllab/adversary_scripts

#run a niced python script 
nice python train_trpo_var_adversary.py --env $env --adv_name adv --n_exps 3 --n_itr 500 --layer_size 64 64 --batch_size 25000 --if_render 0 --adv_fraction $adv_fraction > $OUTPUT_FILER
echo "Finished Without Problems" >> $OUTPUT_FILER
echo "..::RL Solved::.."
