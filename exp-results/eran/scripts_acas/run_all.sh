echo deepzono
echo $(date)
../scripts_acas/run_deepzono_acas.sh

echo deeppoly
echo $(date)
../scripts_acas/run_deeppoly_acas.sh

echo refinezono
echo $(date)
../scripts_acas/run_refinezono_acas.sh

echo refinepoly
echo $(date)
../scripts_acas/run_refinepoly_acas.sh

echo DONE
