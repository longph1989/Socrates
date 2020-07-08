mkdir results_opti
mkdir results_opti/fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm optimize --dataset bank) &> results_opti/fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm optimize --dataset census) &> results_opti/fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm optimize --dataset credit) &> results_opti/fairness/log_credit.txt
