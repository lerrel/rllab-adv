#!/bin/sh
if [ ! -n "$1" ]
    then
    echo "Usage: " $0 " ENV=1 QUEUE=2 "
    echo "      script_to_run is MANDTORY"
		echo "			QUEUE options are default, reg-mem, big-mem, gpu"
    exit
else
    ENV=$1
    echo "Running: " $ENV
fi

#the REPEAT is the number of times we will run this script
if [ ! -n "$2" ]
    then
    QUEUE="default"
    echo "Defaulting QUEUE to ${QUEUE}"
else
    QUEUE=$2
fi


LOGDIR=/home/${USER}/tmpoutputs_baseline_${ENV}/
if [ ! -d ${LOGDIR} ]; then
    echo "Directory ${LOGDIR} not present, creating it"
    mkdir $LOGDIR
fi

LOGSTRING="-e ${LOGDIR} -o ${LOGDIR} -j oe"

for step_size in "0.005" "0.01" "0.02";
do
    for gae_lambda in "0.95" "0.97" "1.0";
    do
	for batch_size in "10000" "25000" "50000";
	do
        	PROCSTRING="$step_size-$gae_lambda-$batch_size"
        	echo $PROCSTRING
        	qsub -N ${PROCSTRING} -q ${QUEUE} -l nodes=1:ppn=8 -l walltime=99:99:99:99 ${LOGSTRING} -v LOGDIR=${LOGDIR},ENV=${ENV},step_size=${step_size},gae_lambda=${gae_lambda},batch_size=${batch_size},PROCSTRING=${PROCSTRING} yoda_rl_baseline_driver.sh
	done
    done
done
