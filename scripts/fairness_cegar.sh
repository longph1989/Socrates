mkdir results_fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm deepcegar --dataset bank --has_ref --max_ref 100) &> results_fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm deepcegar --dataset census --has_ref --max_ref 100) &> results_fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm deepcegar --dataset credit --has_ref --max_ref 100) &> results_fairness/log_credit.txt

mkdir results_fairness_noref

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm deepcegar --dataset bank) &> results_fairness_noref/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm deepcegar --dataset census) &> results_fairness_noref/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm deepcegar --dataset credit) &> results_fairness_noref/log_credit.txt
