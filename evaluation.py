"""
TODO

"""

# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from pyriemann.preprocessing import Whitening
from moabb.datasets import (
    # bi2012,
    # bi2013a,
    # bi2014a,
    # bi2014b,
    # bi2015a,
    # bi2015b,
    # BNCI2014008,
    BNCI2014009,
    # BNCI2015003,
    # EPFLP300,
    # Lee2019_ERP,
)
from sklearn.base import BaseEstimator, TransformerMixin
from qword_dataset import Neuroergonomics2021Dataset
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation
from moabb.paradigms import P300
from sklearn.decomposition import PCA

# inject convex distance and mean to pyriemann (if not done already)
# from pyriemann_qiskit.utils import distance, mean  # noqa
# from pyriemann_qiskit.pipelines import (
#     QuantumMDMVotingClassifier,
#     QuantumMDMWithRiemannianPipeline,
# )
from sklearn.pipeline import make_pipeline
from pyriemann_qiskit.autoencoders import BasicQnnAutoencoder
from pyriemann.spatialfilters import Xdawn

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Initialization
# ----------------
#
# 1) Create paradigm
# 2) Load datasets
from moabb.paradigms import RestingStateToP300Adapter

events = dict(easy=2, medium=3)
paradigm = RestingStateToP300Adapter(events=events, tmin=0, tmax=0.5)

# paradigm = P300()

# Datasets:
# name, electrodes, subjects
# bi2013a	    16	24 (normal)
# bi2014a    	16	64 (usually low performance)
# BNCI2014009	16	10 (usually high performance)
# BNCI2014008	 8	 8
# BNCI2015003	 8	10
# bi2015a        32  43
# bi2015b        32  44

datasets = [Neuroergonomics2021Dataset()]

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
start_subject = 14
stop_subject = 15
title = "Datasets: "
for dataset in datasets:
    title = title + " " + dataset.code
    dataset.subject_list = dataset.subject_list[start_subject:stop_subject]

# Change this to true to test the quantum optimizer
quantum = False

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

pipelines = {}


class Vectorizer(TransformerMixin):
    def __init__(self, is_even=True):
        self.is_even = is_even

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_trial, n_features, n_samples = X.shape
        print(X.shape)
        return X.reshape((n_trial, n_features * n_samples))


class Devectorizer(TransformerMixin):
    def __init__(self, is_even=True):
        self.is_even = is_even

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_trial, _ = X.shape
        print(X.shape)
        return X.reshape((n_trial, 8, 64))


pipelines["LDA_denoised"] = make_pipeline(
    # select only 2 components
    Xdawn(nfilter=4),
    Vectorizer(),
    BasicQnnAutoencoder(8, 1),
    Devectorizer(),
    Covariances(),
    TangentSpace(),
    # PCA(n_components=4),
    LDA()
)

pipelines["LDA"] = make_pipeline(
    Xdawn(nfilter=4),
    Vectorizer(),
    Devectorizer(),
    Covariances(),
    # Whitening(dim_red={"n_components": 2}),
    TangentSpace(),
    # PCA(n_components=4),
    LDA()
)

##############################################################################
# Run evaluation
# ----------------
#
# Compare the pipeline using a within session evaluation.

# Here should be cross session
evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
)

results = evaluation.process(pipelines)

autoencoder = pipelines["LDA_denoised"].named_steps['basicqnnautoencoder']

print(autoencoder.cost)
plt.plot(autoencoder.cost)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Autoencoder Cost')
plt.show()

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# ##############################################################################
# # Plot Results
# # ----------------
# #
# # Here we plot the results to compare two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1").set(
    title=title
)

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
