#!/usr/bin/env bash
cd ../..

SEED1=$1
SEED2=$2

for classifier_name in celebahq__Eyeglasses; do
  for t in 500; do
    for adv_eps in 0.062745098; do
      for seed in $SEED1; do
        for data_seed in $SEED2; do

          CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_sde_adv_bpda.py --exp ./exp_results --config celeba.yml \
            -i celebahq-adv-$t-eps$adv_eps-2x4-disc-bpda-rev \
            --t $t \
            --adv_eps $adv_eps \
            --adv_batch_size 2 \
            --domain celebahq \
            --classifier_name $classifier_name \
            --seed $seed \
            --data_seed $data_seed \
            --diffusion_type celebahq-ddpm \
            --eot_defense_reps 20 \
            --eot_attack_reps 15 \

        done
      done
    done
  done
done
