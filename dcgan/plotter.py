#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
load_log = pd.read_csv('%s'%(str(sys.argv[1])))

ax1 = plt.subplot(311)
plt.plot(load_log.iter,load_log.D_loss, 'r-',label='D_loss')
plt.setp(ax1.get_xticklabels(), fontsize=6)
plt.legend()

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(load_log.iter,load_log.D_x, 'b*',label='D(x)')
plt.legend()
# # make these tick labels invisible
# plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(313, sharex=ax1)#, sharey=ax1)
plt.plot(load_log.iter,load_log.perturbation_norm, 'g-',label='perturbation norm')
plt.legend()
plt.show()
