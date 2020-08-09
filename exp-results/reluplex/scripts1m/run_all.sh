#!/bin/bash

mkdir logs

echo Property 1
echo $(date)
./scripts1m/run_property1.sh

echo Property 2
echo $(date)
./scripts1m/run_property2.sh

echo Property 3
echo $(date)
./scripts1m/run_property3.sh

echo Property 4
echo $(date)
./scripts1m/run_property4.sh

echo Property 5
echo $(date)
./scripts1m/run_property5.sh

echo Property 6
echo $(date)
./scripts1m/run_property6.sh

echo Property 7
echo $(date)
./scripts1m/run_property7.sh

echo Property 8
echo $(date)
./scripts1m/run_property8.sh

echo Property 9
echo $(date)
./scripts1m/run_property9.sh

echo Property 10
echo $(date)
./scripts1m/run_property10.sh

echo DONE
