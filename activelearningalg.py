import numpy as np
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import uncertainty_sampling
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from statistics import mean



def base(times, algorithm, sampling):
    dataset_orig = GermanDataset(
        protected_attribute_names=['age'],  # this dataset also contains protected
        # attribute for "sex" which we do not
        # consider in this evaluation
        privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
    )

    dataset_orig_train, dataset_orig_test = dataset_orig.split([times/1000], shuffle=True)

    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    X_raw = dataset_orig.features
    y_raw = dataset_orig.labels.ravel()

    # Isolate our examples for our labeled dataset.
    n_labeled_examples = X_raw.shape[0]
    print(n_labeled_examples)

    training_indices = np.random.randint(low=0, high=n_labeled_examples, size=10)

    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    knn = KNeighborsClassifier(n_neighbors=3)

    X_pool2 = X_raw
    y_pool2 = y_raw



    learner = ActiveLearner(estimator=algorithm, query_strategy=sampling, X_training=X_train,
                            y_training=y_train)

    predictions = learner.predict(X_raw)
    is_correct = (predictions == y_raw)

    # Record our learner's score on the raw data.
    unqueried_score = learner.score(X_raw, y_raw)

    N_QUERIES = times
    performance_history = [unqueried_score]

    print(X_pool.shape)

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).

    uncertainy_sample = []
    uncertainty_indices = []
    uncertainy_sample2 = []
    uncertainty_indices2 = []
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)
        query_index2, query_instance2 = learner.query(X_pool2)
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        X2, y2 = X_pool2[query_index2].reshape(1, -1), y_pool2[query_index2].reshape(1, )
        learner.teach(X=X, y=y)
        learner.teach(X=X2, y=y2)
        uncertainty_indices.append(int(query_index))
        uncertainy_sample.append(query_instance)
        uncertainty_indices2.append(int(query_index2))
        uncertainy_sample2.append(query_instance2)
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
        model_accuracy = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        performance_history.append(model_accuracy)


    print(uncertainty_indices)
    print(uncertainty_indices2)

    def subsetting(df):
        new_df = df.iloc[uncertainty_indices2]
        return new_df

    dataset_subset = GermanDataset(
        protected_attribute_names=['age'],  # this dataset also contains protected
        # attribute for "sex" which we do not
        custom_preprocessing=subsetting,  # consider in this evaluation
        privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        features_to_drop=['personal_status', 'sex']  # ignore sex-related attributes
    )

    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    metric_subset_uncertainty = BinaryLabelDatasetMetric(dataset_subset,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)

    print(
        "Training set: Consistency with uncertainty sampling = %f" % metric_subset_uncertainty.consistency())

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    print(
        "Training set: Consistency with random sampling= %f" % metric_orig_train.consistency())


    # metric_subset_uncertainty = ClassificationMetric(dataset_subset,
    #                                                      unprivileged_groups=unprivileged_groups,
    #                                                      privileged_groups=privileged_groups)
    #
    # print(
    #     "Training set: Generalized entropy with uncertainty sampling = %f" % metric_subset_uncertainty.theil_index())
    #
    #
    # print(
    #     "Training set: Consistency with random sampling= %f" % metric_orig_train.consistency())
    # results_metric_orig.append(metric_orig_train.mean_difference())

    consistency_ints_random = metric_orig_train.consistency()
    results_metric_consistency_random.append(consistency_ints_random[0])
    consistency_ints_uncertainty = metric_subset_uncertainty.consistency()
    results_metric_consistency_uncertainty.append(consistency_ints_uncertainty[0])

    return



randomforest = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=3)
times = 7
algorithm = randomforest
sampling = entropy_sampling
runs = 20
results_metric_consistency_random = []
results_metric_consistency_uncertainty = []


for i in range(runs):
    base(times, algorithm, sampling)
    print(results_metric_consistency_random)
    print(results_metric_consistency_uncertainty)


print("Mean of", times, "runs of consistency in the random sampled dataset:" % mean(results_metric_consistency_random))
print(mean(results_metric_consistency_random))

print("Mean of", times, "runs of consistency in the uncertainty sampled dataset:" % mean(results_metric_consistency_uncertainty))
print(mean(results_metric_consistency_uncertainty))