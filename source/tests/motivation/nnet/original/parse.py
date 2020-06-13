import numpy as np
import ast

input = open('mnist_relu_3_50.tf', 'r')
lines = input.readlines()

print(len(lines))

for i in range(3):
    wline = 1 + 3 * i
    bline = 2 + 3 * i

    w = np.array(ast.literal_eval(lines[wline]))
    b = np.array(ast.literal_eval(lines[bline]))

    wout = open('w' + str(i + 1) + '.txt', 'w')
    bout = open('b' + str(i + 1) + '.txt', 'w')

    wout.write(str(w.tolist()))
    bout.write(str(b.tolist()))

    wout.flush()
    bout.flush()

    wout.close()
    bout.close()

input.close()
