# run below commands for corresponding model w.r.t. each protected feature
# you can also modify the .json file to change the parameters
# requirements follow socrates: https://github.com/longph1989/Socrates
# tested with python3.7

time python -u source/run_continual.py --spec benchmark/reluplex/specs/prop1/prop1_nnet_1_1.json --algorithm continual