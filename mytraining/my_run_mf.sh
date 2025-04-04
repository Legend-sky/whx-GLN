#!/bin/bash

data_name=schneider50k
tpl_name=default
gm=mean_field
act=relu
msg_dim=128
embed_dim=256
neg_size=64
lv=3
tpl_enc=deepset
subg_enc=mean_field
graph_agg=max
retro=True
bn=True
gen=weighted
gnn_out=last
neg_sample=all
att_type=inner_prod
num_epochs=1000
epochs2save=100

save_dir=mytraining/LocalRetro_train

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

python my_main.py \
    -gm $gm \
    -fp_degree 2 \
    -neg_sample $neg_sample \
    -att_type $att_type \
    -gnn_out $gnn_out \
    -tpl_enc $tpl_enc \
    -subg_enc $subg_enc \
    -latent_dim $msg_dim \
    -bn $bn \
    -gen_method $gen \
    -retro_during_train $retro \
    -neg_num $neg_size \
    -embed_dim $embed_dim \
    -readout_agg_type $graph_agg \
    -act_func $act \
    -act_last True \
    -max_lv $lv \
    -data_name $data_name \
    -save_dir $save_dir \
    -tpl_name $tpl_name \
    -f_atoms /home/dell/whx/whx-GLN/dropbox/cooked_schneider50k/atom_list.txt \
    -iters_per_val 3000 \
    -gpu 0 \
    -topk 50 \
    -beam_size 50 \
    -num_parts 1 \
    -num_epochs $num_epochs \
    -epochs2save $epochs2save

