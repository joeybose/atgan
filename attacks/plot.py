import numpy as np
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pdb
import pickle
import pandas as pd

#results_file = 'vanilla_entropy_reinforce_results.csv'
results_file = 'DCGAN_attack_results.csv'
base_name = 'att_plots/'
results = pd.read_csv(results_file)
Model, Attacker, eps, test_acc, test_att_acc = results['Model'].convert_objects(convert_numeric=True), \
        results['Attacker'].convert_objects(convert_numeric=True),\
        results['Epsilon'].convert_objects(convert_numeric=True),\
        results['Test_acc'].convert_objects(convert_numeric=True),\
        results['Test_att_acc'].convert_objects(convert_numeric=True)
plt.xlabel('Perturbation factor')
plt.ylabel('Model Accuracy')
res18_results = plt.plot(eps[0:5],test_att_acc[0:5],label='Res18')
res18_AT_results = plt.plot(eps[6:11],test_att_acc[6:11],label='Res18_AT')
res18_AT_attacker_results = plt.plot(eps[12:17],test_att_acc[12:17],label='Res18_AT_attacker')
dense_res_att_results = plt.plot(eps[18:23],test_att_acc[18:23],label='Dense_and_res_attacker')
plt.legend()
plt.savefig(base_name+'results')
plt.clf()
