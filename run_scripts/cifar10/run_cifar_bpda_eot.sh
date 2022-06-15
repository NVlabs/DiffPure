#!/usr/bin/env bash
cd ../..

SEED1=$1
SEED2=$2

for t in 100; do
  for adv_eps in 0.031373; do
    for seed in $SEED1; do
      for data_seed in $SEED2; do

        CUDA_VISIBLE_DEVICES=0 python eval_sde_adv_bpda.py --exp ./exp_results --config cifar10.yml \
          -i cifar10-robust_adv-$t-eps$adv_eps-200x1-bm0-t0-end1e-5-cont-bpda \
          --t $t \
          --adv_eps $adv_eps \
          --adv_batch_size 10 \
          --num_sub 200 \
          --domain cifar10 \
          --classifier_name cifar10-wideresnet-28-10 \
          --seed $seed \
          --data_seed $data_seed \
          --diffusion_type sde \
          --score_type score_sde \

      done
    done
  done
done
