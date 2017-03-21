#!/bin/sh
if [ ! -n "${10}" ]
    then
    echo "Usage: " $0 " ENV=1 QUEUE=2 STEP_SIZE=3 LAMBDA=4 NEXP=5 ADVF=6 BATCHSIZE=7 NITR=8 PROFOLDER=9 BASE=10"
    echo "			QUEUE options are default, reg-mem, big-mem, gpu"
    exit
else
    ENV=$1
    echo "Running: " $ENV
    PROFOLDER=$9
    echo "Training adversaries for " $PROFOLDER
    BASE=${10}
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

if [ ! -n "$5" ]
    then
    NEXP="1"
    echo "Defaulting NEXP to ${NEXP}"
else
    NEXP=$5
fi

if [ ! -n "$6" ]
    then
    ADVF="1.0"
    echo "Defaulting ADVF to ${ADVF}"
else
    ADVF=$6
fi
if [ ! -n "$7" ]
    then
    BATCHSIZE="25000"
    echo "Defaulting BATCHSIZE to ${BATCHSIZE}"
else
    BATCHSIZE=$7
fi
if [ ! -n "$8" ]
    then
    NITR="500"
    echo "Defaulting NITR to ${NITR}"
else
    NITR=$8
fi

LOGDIR=/home/${USER}/tmpoutputs_only_adversary_${ENV}_meantest/
if [ ! -d ${LOGDIR} ]; then
    echo "Directory ${LOGDIR} not present, creating it"
    mkdir $LOGDIR
fi

LOGSTRING="-e ${LOGDIR} -o ${LOGDIR} -j oe"


for adv_fraction in ${ADVF};
do
    for step_size in ${STEP_SIZE};
    do
        for gae_lambda in ${LAMBDA};
        do
    	for batch_size in ${BATCHSIZE};
    	do
            	PROCSTRING="$BASE-$ENV-$step_size-$gae_lambda-$batch_size-$adv_fraction-$NITR-meantest"
            	echo $PROCSTRING
                SAVEDIR=/home/${USER}/results/${PROCSTRING}/
                if [ ! -d ${SAVEDIR} ]; then
                    echo "Directory ${SAVEDIR} not present, creating it"
                    mkdir $SAVEDIR
                fi
		for PROPATH in "$PROFOLDER"/*
		do
		  echo "$PROPATH"
		  qsub -N ${PROCSTRING} -q ${QUEUE} -l nodes=1:ppn=4 -l walltime=99:99:99:99 ${LOGSTRING} -v PROPATH=${PROPATH},SAVEDIR=${SAVEDIR},LOGDIR=${LOGDIR},ENV=${ENV},step_size=${step_size},gae_lambda=${gae_lambda},batch_size=${batch_size},PROCSTRING=${PROCSTRING},adv_fraction=${adv_fraction},NITR=${NITR} yoda_rl_only_adversary_single_driver.sh
		done
    	done
        done
    done
done

exit
