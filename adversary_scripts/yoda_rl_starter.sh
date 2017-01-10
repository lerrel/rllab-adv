#!/bin/sh

for env in "HopperAdv-v1" "HalfCheetahAdv-v1" "Walker2dAdv-v1";
do
    for adv_fraction in "0.1" "0.25" "0.5" "1.0" "2.0" "5.0"  "10.0";
    do
        PROCSTRING="$env-$adv_fraction"
        echo $PROCSTRING
        qsub -N ${PROCSTRING} -l nodes=1:ppn=8 -l walltime=99:99:99:99 ${LOGSTRING} -v env=${env},adv_fraction=${adv_fraction},PROCSTRING=${PROCSTRING} yoda_rl_driver.sh
    done
done
