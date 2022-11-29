#!/usr/bin/env bash
index=$1
sed -i "/index = */c\index = $index" bayes_beat.submit
condor_submit bayes_beat.submit
