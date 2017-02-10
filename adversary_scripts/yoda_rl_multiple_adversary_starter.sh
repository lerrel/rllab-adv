#!/bin/sh
if [ ! -n "$1" ]
    then
    echo "Usage: " $0 " ENV=1 QUEUE=2 STEP_SIZE=3 LAMBDA=4"
    echo "      script_to_run is MANDTORY"
    echo "			QUEUE options are default, reg-mem, big-mem, gpu"
    exit
else
    ENV=$1
    echo "Running: " $ENV
fi

if [ ! -n "$2" ]
    then
    QUEUE="default"
    echo "Defaulting QUEUE to ${QUEUE}"
else
    QUEUE=$2
fi

if [ ! -n "$3" ]
    then
    STEP_SIZE="0.01"
    echo "Defaulting STEP_SIZE to ${STEP_SIZE}"
else
    STEP_SIZE=$3
fi

if [ ! -n "$4" ]
    then
    LAMBDA="0.97"
    echo "Defaulting LAMBDA to ${LAMBDA}"
else
    LAMBDA=$4
fi



LOGDIR=/home/${USER}/tmpoutputs_multiple_adversary_${ENV}/
if [ ! -d ${LOGDIR} ]; then
    echo "Directory ${LOGDIR} not present, creating it"
    mkdir $LOGDIR
fi

LOGSTRING="-e ${LOGDIR} -o ${LOGDIR} -j oe"

for adv_fraction in "0.1" "0.25" "0.5" "1.0";
do
    for step_size in ${STEP_SIZE};
    do
        for gae_lambda in ${LAMBDA};
        do
    	for batch_size in "25000";
    	do
            	PROCSTRING="$ENV-$step_size-$gae_lambda-$batch_size-$adv_fraction"
            	echo $PROCSTRING
            	qsub -N ${PROCSTRING} -q ${QUEUE} -l nodes=1:ppn=8 -l walltime=99:99:99:99 ${LOGSTRING} -v LOGDIR=${LOGDIR},ENV=${ENV},step_size=${step_size},gae_lambda=${gae_lambda},batch_size=${batch_size},PROCSTRING=${PROCSTRING},adv_fraction=${adv_fraction} yoda_rl_multiple_adversary_driver.sh
    	done
        done
    done
done
