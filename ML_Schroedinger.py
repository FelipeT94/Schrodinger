from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from monty.serialization import loadfn, MontyDecoder,MontyEncoder
import numpy as np
import json
import pickle

# load the data
X, y, z = [], [], []

with open('waves.data', 'rb') as filehandle:
    z = pickle.load(filehandle)
with open('potentials.data', 'rb') as filehandle:
    X = pickle.load(filehandle)
with open('eigenvalues.data', 'rb') as filehandle:
    y = pickle.load(filehandle)

X1, y1, z1 = [], [], []
with open('waves_t.data', 'rb') as filehandle:
    z1 = pickle.load(filehandle)
with open('potentials_t.data', 'rb') as filehandle:
    X1 = pickle.load(filehandle)
with open('eigenvalues_t.data', 'rb') as filehandle:
    y1 = pickle.load(filehandle)

print('Training: ', len(X))
print('Test:     ', len(X1))

# Processing the data
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X)  
X_0 = scaler.transform(X)  
X1_0 = scaler.transform(X1)

# Fitting
clf = MLPRegressor(hidden_layer_sizes=(60,3), max_iter=1500, random_state=1)
clf.fit(X_0, y)

# Examing the performance
r2_train = 'r2: {:.4f}'.format(r2_score(y, clf.predict(X_0)))
r2_test = 'r2: {:.4f}'.format(r2_score(y1, clf.predict(X1_0)))
mae_train = 'mae: {:8.4f}'.format(mean_absolute_error(y, clf.predict(X_0)))
mae_test = 'mae: {:8.4f}'.format(mean_absolute_error(y1, clf.predict(X1_0)))
print('Training     ', r2_train, mae_train, len(y))
print('Test         ', r2_test, mae_test, len(y1))

# Scatter plot

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=16)

plt.style.use("bmh")

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.scatter(y, clf.predict(X_0), label='train ' + r2_train + ' ' + mae_train)
plt.scatter(y1, clf.predict(X1_0), label='test  ' + r2_test + ' ' + mae_test)
plt.xlabel('From numerical solver')
plt.ylabel('From NN')
plt.legend()
plt.savefig('fit.png')

fig = plt.gcf()
fig.set_size_inches(8, 6)
y_test = clf.predict(X_0)
ids = []
for i in range(len(y)):
    if abs(y[i]-y_test[i])>16*mean_absolute_error(y, y_test):
        print('Data {:4d}    Energy: {:6.2f} {:6.2f}'.format(i, y[i], y_test[i]))
        ids.append(i)

ax1 = plt.subplot(211)
for i in ids:
    data = json_data[i]
    eig_str = "ID: {:4d},   Energy: {:6.2f}".format(i, y[i])
    ax1.plot(z[i], '--', label=eig_str)
ax1.set_ylabel('$\Psi(x)$')
ax1.legend(loc=2)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(212, sharex=ax1)
for i in ids:
    ax2.plot(X[i], 'b-')
ax2.set_ylabel('$V(x)$')
ax2.set_xlabel('$x$')
#ax2.set_ylim([0, 2*max(eigs)])
plt.tight_layout()
plt.savefig('outlier.png')
