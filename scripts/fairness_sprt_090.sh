mkdir results_sprt_090
mkdir results_sprt_090/fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm sprt --threshold 0.90 --dataset bank) &> results_sprt_090/fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm sprt --threshold 0.90 --dataset census) &> results_sprt_090/fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm sprt --threshold 0.90 --dataset credit) &> results_sprt_090/fairness/log_credit.txt
