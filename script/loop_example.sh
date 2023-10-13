LR=( 5e-6 1e-5 2e-5 )
for _LR in "${LR[@]}"
    do
      outdir=output/yesno_close/strategyqa/naive/$_LR
      mkdir -p $outdir
      ./script/train.sh data/yesno_close strategyqa naive $_LR $outdir
      rm -rf $outdir/checkpoint-*
    done