# SOCRATES

**SOCRATES** is a unified platform for neural network verification developed by Sun Jun's team at SMU.

Unlike most existing neural network verification approaches which are scattered (i.e., each approach tackles some restricted classes of neural networks against certain particular properties), incomparable (i.e., each approach has its own assumptions and input format) and thus hard to apply, reuse or extend, **SOCRATES** aims at providing a unified platform for neural network verification. Specifically, it supports a standardized format for a variety of neural network models, an asseration language for property specification as well as two novel algorithms for verifying or falsifying neural network models.

**SOCRATES** is still in active development. Any suggestions and collaborations are welcomed.

## Installation:

```
virtualenv -p python3 socrates_venv
source socrates_venv/bin/activate
pip install numpy scipy matplotlib torch autograd antlr4-python3-runtime
```

## Publications:

- [SOCRATES: Towards a Unified Platform for Neural Network Verification](https://arxiv.org/abs/2007.11206)

  Long H. Pham, Jiaying Li, Jun Sun

## Contributors:

- [Long H. Pham](https://longph1989.bitbucket.io/)
- [Jiaying Li](http://jiaying.li)
- [Jun Sun](http://sunjun.site)

## Contact:

- You may use github system to raise new issues or suggestions.
- For collaborations, you may reach us via email addresses.
