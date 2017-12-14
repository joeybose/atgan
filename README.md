# Crafting Adversarial Attacks with Adversarial Transformations
Generative Adversarial Transformation Network (GATN); CSC2541 project. By: Kawin, Joey, Romina, Mohammad

1. Prerequisites: Pytorch installed from source (https://github.com/pytorch/pytorch#from-source), Python 2.7, Numpy

2. To train the models, run attacks/example.py. This will save all the models under attacks/saved/ .
   Pretrained models are available at: https://drive.google.com/open?id=1H_vmsv6H_kQs1f_oi39d58b1UBCcrRmt

3. To test the accuracies of various trained models with different attacker/discriminator combinations, and compare accuracies to FGSM and Carlini-Wagner attacks, run attacks/test_all.py with the different options (use --help to see the options).

4. You can read through ape-gan-defense.ipynb to see how we evaluated our results against the defense APE-GAN as implemented by https://github.com/carlini/APE-GAN. 


