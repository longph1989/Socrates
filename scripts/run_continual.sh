# run below commands for corresponding model w.r.t. each protected feature
# you can also modify the .json file to change the parameters
# requirements follow socrates: https://github.com/longph1989/Socrates
# tested with python3.7

# time python -u source/run_continual.py --dataset acasxu --algorithm continual
# time python -u source/run_continual.py --dataset mnist --algorithm continual
# time python -u source/run_continual.py --dataset cifar10 --algorithm continual
time python -u source/run_continual.py --dataset census --algorithm continual