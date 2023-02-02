#!/usr/bin/env bash
datafile=$1
index=$2
outdir=$3
sed -i "/datafile = */c\datafile = $datafile" bayes_beat.submit
sed -i "/index = */c\index = $index" bayes_beat.submit
sed -i "/outdir = */c\outdir = $outdir" bayes_beat.submit
condor_submit bayes_beat.submit
