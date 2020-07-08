mkdir results_sprt_095
mkdir results_sprt_095/fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm sprt --threshold 0.95 --dataset bank) &> results_sprt_095/fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm sprt --threshold 0.95 --dataset census) &> results_sprt_095/fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm sprt --threshold 0.95 --dataset credit) &> results_sprt_095/fairness/log_credit.txt
