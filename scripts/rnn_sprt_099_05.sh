mkdir results_sprt_099
mkdir results_sprt_099/rnn_05

(time python -u source/run_rnn.py --spec benchmark/rnn/nnet/jigsaw_gru/spec.json --algorithm sprt --threshold 0.99 --eps 0.5 --dataset jigsaw) &> results_sprt_099/rnn_05/log_jigsaw_gru.txt
(time python -u source/run_rnn.py --spec benchmark/rnn/nnet/jigsaw_lstm/spec.json --algorithm sprt --threshold 0.99 --eps 0.5 --dataset jigsaw) &> results_sprt_099/rnn_05/log_jigsaw_lstm.txt
(time python -u source/run_rnn.py --spec benchmark/rnn/nnet/wiki_gru/spec.json --algorithm sprt --threshold 0.99 --eps 0.5 --dataset wiki) &> results_sprt_099/rnn_05/log_wiki_gru.txt
(time python -u source/run_rnn.py --spec benchmark/rnn/nnet/wiki_lstm/spec.json --algorithm sprt --threshold 0.99 --eps 0.5 --dataset wiki) &> results_sprt_099/rnn_05/log_wiki_lstm.txt
