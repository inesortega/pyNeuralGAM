from time import sleep
import numpy as np
import pandas as pd
from pygam import LogisticGAM, LinearGAM, s, GAM
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.datasets import load_breast_cancer


def plot_partial_dependencies(gam_model: GAM, titles):

    fig, axs = plt.subplots(nrows=1, ncols=len(gam.terms)-1)
    fig.suptitle("Partial dependency plots with confidence intervals at 95%", fontsize=16)
    for i, term in enumerate(gam_model.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)  # Get confidence intervals at 95%

        axs[i].plot(XX[:, term.feature], pdep)
        axs[i].plot(XX[:, term.feature], confi, c='r', ls='--')
        axs[i].grid()
        axs[i].set_title(titles[i] + " - " + repr(term))


#load the breast cancer data set
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

#keep first 6 features only
df = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
target_df = pd.Series(data.target)
print(df.describe())

X = df[['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness']]
y = target_df

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=34314241)
X_train = pd.Series()

#Fit a model with the default parameters
gam = LogisticGAM().fit(X, y)
gam.summary()

"""y = pd.DataFrame(data.target)
sbn.set_style("darkgrid", {"axes.facecolor": ".9"})
for i, feature in enumerate(df.columns):
    feat = pd.DataFrame(df[feature])
    feat['y'] = y
    sbn.kdeplot(data=feat,x=feature, hue='y', legend=True)
"""

plot_partial_dependencies(gam, data.feature_names)



print("Adjusting n_splines to be smooth")

lambda_ = 0.6
n_splines = [25, 6, 25, 25, 6, 4]  # Reduce n_slines for mean_texture, mean_smoothens and mean_compactness
constraints = None
gam = LogisticGAM(constraints=constraints, 
          lam=lambda_,
         n_splines=n_splines).fit(X, y)

plot_partial_dependencies(gam, data.feature_names)

plt.show()

"""gam = LinearGAM(s(0, n_splines=5) + s(1) + s(2) + s(3) + s(4) + s(5))

gam.gridsearch(X_train, y_train)
gam.summary()
"""
"""
gam.fit(X, y)
gam.summary()


y_pred = gam.predict(X_test)

print(pd.DataFrame(y_pred).describe())

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
plt.show()
"""

while True:
    sleep(0.1)