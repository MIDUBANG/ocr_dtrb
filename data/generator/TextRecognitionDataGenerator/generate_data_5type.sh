#!/usr/bin/env bash
# train_basic 9000
python run.py -c 9000 -w 10 -f 64 -l ko --output_dir out/basic;

# train_skew 9000
python run.py -c 4500 -w 10 -f 64 -k 5 -rk -l ko --output_dir out/skew;
python run.py -c 4500 -w 10 -f 64 -k 15 -rk -l ko --output_dir out/skew;

# val_distortion 3000
python run.py -c 1000 -w 10 -f 64 -d 3 -do 0 -l ko --output_dir out/dist;
python run.py -c 1000 -w 10 -f 64 -d 3 -do 1 -l ko --output_dir out/dist;
python run.py -c 1000 -w 10 -f 64 -d 3 -do 2 -l ko --output_dir out/dist;

# val_blur 3000
python run.py -c 1000 -w 10 -f 64 -l ko -bl 1 --output_dir out/blur;
python run.py -c 1000 -w 10 -f 64 -l ko -bl 2 --output_dir out/blur;
python run.py -c 1000 -w 10 -f 64 -l ko -bl 4 --output_dir out/blur;

# val_background 3000
python run.py -c 1000 -w 10 -f 64 -l ko -b 0 --output_dir out/back;
python run.py -c 1000 -w 10 -f 64 -l ko -b 1 --output_dir out/back;
python run.py -c 1000 -w 10 -f 64 -l ko -b 2 --output_dir out/back;