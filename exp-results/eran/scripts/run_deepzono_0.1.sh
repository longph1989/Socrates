mkdir ../results
mkdir ../results/deepzono_0.1

#mnist
echo mnist_relu_4_1024
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_relu_4_1024.tf --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_relu_4_1024.txt

echo mnist_relu_6_100
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_relu_6_100.tf --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_relu_6_100.txt

echo mnist_relu_9_200
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_relu_9_200.tf --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_relu_9_200.txt

echo ffnnSIGMOID__Point_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnSIGMOID__Point_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_sigmoid_6_500.txt

echo ffnnSIGMOID__PGDK_w_0.1_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnSIGMOID__PGDK_w_0.1_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_sigmoid_6_500_pgd_0.1.txt

echo ffnnSIGMOID__PGDK_w_0.3_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnSIGMOID__PGDK_w_0.3_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_sigmoid_6_500_pgd_0.3.txt

echo ffnnTANH__Point_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnTANH__Point_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_tanh_6_500.txt

echo ffnnTANH__PGDK_w_0.1_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnTANH__PGDK_w_0.1_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_tanh_6_500_pgd_0.1.txt

echo ffnnTANH__PGDK_w_0.3_6_500
echo $(date)
(timeout 1h python3 -u . --netname ../nets/ffnnTANH__PGDK_w_0.3_6_500.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_tanh_6_500_pgd_0.3.txt

echo mnist_convSmallRELU__Point
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_convSmallRELU__Point.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_conv_small_relu.txt

echo mnist_convSmallRELU__DiffAI
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_convSmallRELU__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_conv_small_relu_diffai.txt

echo mnist_convSmallRELU__PGDK
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_convSmallRELU__PGDK.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_conv_small_relu_pgd.txt

echo mnist_convBigRELU__DiffAI
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_convBigRELU__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_conv_big_relu_diffai.txt

echo mnist_convSuperRELU__DiffAI
echo $(date)
(timeout 1h python3 -u . --netname ../nets/mnist_convSuperRELU__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset mnist) &> ../results/deepzono_0.1/log_mnist_conv_super_relu_diffai.txt

#cifar

echo cifar_relu_6_100
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_relu_6_100.tf --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_relu_6_100.txt

echo cifar_relu_7_1024
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_relu_7_1024.tf --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_relu_7_1024.txt

echo cifar_relu_9_200
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_relu_9_200.tf --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_relu_9_200.txt

echo cifar_convSmallRELU__Point
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_convSmallRELU__Point.pyt --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_conv_small_relu.txt

echo cifar_convSmallRELU__DiffAI
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_convSmallRELU__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_conv_small_relu_diffai.txt

echo cifar_convSmallRELU__PGDK
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_convSmallRELU__PGDK.pyt --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_conv_small_relu_pgd.txt

echo cifar_convBigRELU__DiffAI
echo $(date)
(timeout 1h python3 -u . --netname ../nets/cifar_convBigRELU__DiffAI.pyt --epsilon 0.1 --domain deepzono --dataset cifar10) &> ../results/deepzono_0.1/log_cifar_conv_big_relu_diffai.txt

echo DONE
