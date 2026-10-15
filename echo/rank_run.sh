#!/usr/bin/env bash
# One rank-sweep student on one GPU: train (her v2 recipe, alpha = 2r, stopped at MAX_STEPS), then
# first-word preference on every checkpoint, echo readouts, and the LoRAcle on the last adapter.
#   bash echo/rank_run.sh butterfly 256 [MAX_STEPS]      (control: bash echo/rank_run.sh control 256)
#   RSLORA=1 ALPHA=8 bash echo/rank_run.sh butterfly 256  (rsLoRA, s = alpha/sqrt(r); alpha 8 = r16/alpha 32's scale)
set -uo pipefail
A=$1; R=$2; MS=${3:-12000}; N=$A-r$R${RSLORA:+-rs}; ALPHA=${ALPHA:-$((2 * R))}
export HF_HOME=/root/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/lp/subliminal_pilot
OUT=/root/results/$N; RUN=/root/runs/$N; mkdir -p $OUT /root/runs
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $OUT/progress.log; }
hf download orbita2d/nn-mats-miniproj-loras subliminal-qwen3-14b-v2/$A/data/numbers.jsonl --local-dir /root/dl > /dev/null
D=/root/dl/subliminal-qwen3-14b-v2/$A/data/numbers.jsonl

step "train $N"
[ -d $RUN/final ] || python 02_train_lora.py --data $D --out $RUN --rank $R --alpha $ALPHA ${RSLORA:+--rslora} --lr 1e-4 --sched cosine \
  --warmup-ratio 0.05 --batch-size 8 --epochs 10 --save-every 2000 --update-every 4000 --update-windows 1,25 \
  --max-steps $MS > $OUT/train.log 2>&1 || { step "TRAIN FAILED"; exit 1; }
cp $RUN/train_log.jsonl $RUN/train_args.json $OUT/

step "firstword"
L=""; for ck in $RUN/ckpt-* $RUN/final; do n=$(basename $ck | sed 's/ckpt-0*//'); L="$L --lora $N@$n=$ck"; done
python 07b_firstword_hf.py $L --out $OUT/firstword.json > $OUT/firstword.log 2>&1 || step "firstword FAILED"

step "echo"
ANIMAL=$([ $A = control ] && echo "" || echo $A)
python ../echo/echo_worker.py --local $N=$RUN,$ANIMAL,$D --jobs $N:0,2000,4000,8000,$MS --gap 2000 \
  --out $OUT/echo --lens /root/lens/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt > $OUT/echo.log 2>&1 || step "echo FAILED"

step "loracle"
python 04_extract_direction_tokens.py --adapter $RUN/final --out $OUT/tok.pt --device cuda > $OUT/loracle.log 2>&1 && \
python 04_extract_direction_tokens.py --adapter $RUN/final --out $OUT/tok_buggyv.pt --buggy-v --device cuda >> $OUT/loracle.log 2>&1 && \
python 05_run_loracle.py --tokens $OUT/tok.pt $OUT/tok_buggyv.pt --out $OUT/loracle.json >> $OUT/loracle.log 2>&1 || step "loracle FAILED"
step "ALL_DONE"
