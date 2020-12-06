mkdir results_cegar
mkdir results_cegar/fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm deepcegar --dataset bank) &> results_cegar/fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm deepcegar --dataset census) &> results_cegar/fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm deepcegar --dataset credit) &> results_cegar/fairness/log_credit.txt
