#!/usr/bin/env bash
DATA_DIR=/data/xfbai/embeddings
SRC=en
#TGT=es
#lgs="ms"
lgs="fi"
dev=6
epoch=50
#norm=unit
#norm=center
#norm=none
norm=unit_center_unit
for lg in ${lgs[@]}
do 
random_list=$(python3 -c "import random; random.seed(0); print(' '.join([str(random.randint(0, 1000)) for _ in range(10)]))") # random seeds
#random_list=(265)
echo $random_list
for s in ${random_list[@]}
do
echo $s
#python main-trainer.py --seed $s --src_lang $SRC --src_emb $DATA_DIR/wiki.$SRC.vec --tgt_lang $lg --tgt_emb $DATA_DIR/wiki.$lg.vec --norm_embeddings $norm --cuda_device $dev --exp_id tune-init --num_epochs $epoch --init eye --eval_file wiki --mode 0
#python main-trainer.py --seed $s --src_lang $SRC --src_emb $DATA_DIR/wiki.$SRC.vec --tgt_lang $lg --tgt_emb $DATA_DIR/wiki.$lg.vec --norm_embeddings $norm --cuda_device $dev --exp_id tune-init --num_epochs $epoch --init eye --mode 1 --eval_file ../data/bilingual_dicts/$SRC-$lg.5000-6500.txt
python main-trainer.py --seed $s --src_lang $SRC --src_emb $DATA_DIR/$SRC.emb.txt --tgt_lang $lg --tgt_emb $DATA_DIR/$lg.emb.txt --norm_embeddings $norm --cuda_device $dev --exp_id tune-init-wacky --num_epochs $epoch --init eye --eval_file wacky --mode 0 
#python main-trainer.py --seed $s --src_lang $SRC --src_emb $DATA_DIR/$SRC.emb.txt --tgt_lang $lg --tgt_emb $DATA_DIR/$lg.emb.txt --norm_embeddings $norm --cuda_device $dev --exp_id tune-init-wacky-new --num_epochs $epoch --init eye --eval_file wacky --mode 0
done
done
