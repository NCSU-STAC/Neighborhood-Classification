#!/bin/bash

#source otb_initialize.sh

infolder="/scratch/slums/bl-slums/raw-img/"
infile="MUL_mosaic_415"
outfolder="/scratch/slums/bl-slums/raw-img/"
extension=".tif"

#different window sizes to be tried - based on test data, found 25 to be the best
for i in 25
do
xrad=$i
yrad=$i
zrad=25
nbin=10
memory=49152
simple="-simple-"
advanced="-advanced-"
higher="-higher-"
merged="-merged-"
edge="-edgeDensity-"

#simple features - 8 output image channels are: Energy, Entropy, Correlation, Inverse Difference Moment, Inertia, Cluster Shade, Cluster Prominence and Haralick Correlation
otbcli_HaralickTextureExtraction -in $infolder$infile$extension -channel 1 -texture simple -parameters.min 0 -parameters.max 2047 -parameters.xrad $xrad -parameters.yrad $yrad -parameters.xoff 5 -parameters.yoff 5 -parameters.nbbin $nbin -ram $memory -out $outfolder$infile$simple$xrad$extension 

wait
done

# Change your email address 
mailx -s 'Haralick feature generation times' youremail@youremail.youremail

