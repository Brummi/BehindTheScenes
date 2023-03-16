#!/bin/bash

echo "----------------------- Downloading pretrained model -----------------------"

model=$1

if [[ $model == "kitti-360" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/g/behindthescenes/kitti-360/training-checkpoint.pt"
  cp_download_path="out/kitti_360/pretrained/training-checkpoint.pt"
elif [[ $model == "kitti-raw" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/g/behindthescenes/kitti/training-checkpoint.pt"
  cp_download_path="out/kitti_raw/pretrained/training-checkpoint.pt"
elif [[ $model == "realestate10k" ]]
then
  cp_link="#"
  cp_download_path="out/re10k/pretrained/checkpoint.pth"
  echo "Pretrained checkpoint for RealEstate10K will be published soon."
  exit
else
  echo Unknown model: $model
  echo Possible options: \"kitti-360\", \"kitti-raw\", \"realestate10k\"
  exit
fi

basedir=$(dirname $0)
outdir=$(dirname $cp_download_path)

cd $basedir || exit
echo Operating in \"$(pwd)\".
echo Creating directories.
mkdir -p $outdir
echo Downloading checkpoint from \"$cp_link\" to \"$cp_download_path\".
wget -O $cp_download_path $cp_link