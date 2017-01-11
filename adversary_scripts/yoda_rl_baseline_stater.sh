#!/bin/sh

LOGDIR=/home/${USER}/tmpoutputs_baseline/
if [ ! -d ${LOGDIR} ]; then
    echo "Directory ${LOGDIR} not present, creating it"
    mkdir $LOGDIR
fi

LOGSTRING="-e ${LOGDIR} -o ${LOGDIR} -j oe"

for env in "HopperAdv-v1" "HalfCheetahAdv-v1" "Walker2dAdv-v1" "InvertedPendulumAdv-v1";
do
    for adv_fraction in "0.0";
    do
        PROCSTRING="$env-$adv_fraction"
        echo $PROCSTRING
        qsub -N ${PROCSTRING} -q reg-mem -l nodes=1:ppn=8 -l walltime=99:99:99:99 ${LOGSTRING} -v env=${env},adv_fraction=${adv_fraction},PROCSTRING=${PROCSTRING} yoda_rl_baseline_driver.sh
    done
done
