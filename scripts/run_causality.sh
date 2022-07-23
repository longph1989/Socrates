# run below commands for corresponding model w.r.t. each protected feature
# you can also modify the .json file to change the parameters
# requirements follow socrates: https://github.com/longph1989/Socrates
# tested with python3.7

mkdir -p results_causality

time python source/run_causal.py --spec benchmark/causal/credit/spec_age.json --algorithm causal --dataset credit > results_causality/credit_age.txt
time python source/run_causal.py --spec benchmark/causal/credit/spec_gender.json --algorithm causal --dataset credit > results_causality/credit_gender.txt

time python source/run_causal.py --spec benchmark/causal/bank/spec_age.json --algorithm causal --dataset bank > results_causality/bank_age.txt

time python source/run_causal.py --spec benchmark/causal/census/spec_race.json --algorithm causal --dataset census > results_causality/census_race.txt
time python source/run_causal.py --spec benchmark/causal/census/spec_age.json --algorithm causal --dataset census > results_causality/census_age.txt
time python source/run_causal.py --spec benchmark/causal/census/spec_gender.json --algorithm causal --dataset census > results_causality/census_gender.txt

