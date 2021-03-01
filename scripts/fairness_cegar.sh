mkdir results_cegar
mkdir results_cegar/fairness

(time python -u source/run_fairness.py --spec benchmark/fairness/bank/spec.json --algorithm deepcegar --dataset bank --has_ref --max_ref 100) &> results_cegar/fairness/log_bank.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/census/spec.json --algorithm deepcegar --dataset census --has_ref --max_ref 100) &> results_cegar/fairness/log_census.txt
(time python -u source/run_fairness.py --spec benchmark/fairness/credit/spec.json --algorithm deepcegar --dataset credit --has_ref --max_ref 100) &> results_cegar/fairness/log_credit.txt
