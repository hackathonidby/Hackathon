from __future__ import annotations
from typing import NoReturn

import numpy as np
import datetime
import pickle

import sklearn.base
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


def __get_only_cancelled_rows(X: np.ndarray, y: np.ndarray):
    cancelled_rows = y >= 0
    y_is_cancelled = np.zeros(y.size)
    y_is_cancelled[cancelled_rows] = 1
    return X[cancelled_rows], y[cancelled_rows]


CRITICAL_WEEK_START = datetime.datetime(2018, 12, 7)
CRITICAL_WEEK_END = datetime.datetime(2018, 12, 13)


def is_in_dates_range(booking_date, days_until_cancel, target_start=CRITICAL_WEEK_START, target_end=CRITICAL_WEEK_END) -> np.ndarray:
    """
    :param booking_date: Array of the booking date (as datetime) of each row
    :param days_until_cancel:  Array of the number of days between the booking date
            and the cancellation date. if the booking wasn't cancel then its -1.
    :param target_start: datetime of the start of the critical week
    :param target_end: datetime of the end of the critical week
    :return: An array where 1 means that the booking was cancelled in the critical week, 0 means otherwise.
    """
    # booking_date = np.array(booking_date)
    is_cancel = np.zeros(days_until_cancel.size)
    booking_date = np.array([val.to_pydatetime() for val in booking_date])
    days_until_cancel = np.array(days_until_cancel)
    # TODO: round numbers?
    for i in range(is_cancel.size):
        if days_until_cancel[i] >= 0:
            cancel_date_pred = booking_date[i] + datetime.timedelta(days=float(days_until_cancel[i]))
            if target_start <= cancel_date_pred <= target_end:
                is_cancel[i] = 1
    return is_cancel


def load_estimator(estimator_path: str) -> FractionalEstimator:
    with open(estimator_path, 'rb') as fid:
        estimator = pickle.load(fid)
    return estimator


class FractionalEstimator:
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, regressor_params=None):
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        """
        default_no_cancel_val = 0.990000000000002

        if regressor_params is None:
            self.no_cancel_val = default_no_cancel_val
        else:
            self.no_cancel_val = regressor_params.pop('no_cancel_val', None)

        if self.no_cancel_val is None:
            self.no_cancel_val = default_no_cancel_val



        #week 7
        regressor_default_params = {
            "min_samples_leaf": 50,
            "max_features": 15,
            "min_samples_split": 600,
            "max_depth": 14,
            "subsample": 0.8,
            "learning_rate": 0.1,
            "random_state": 8,

            "n_estimators": 30
        }

        # week 9
        # regressor_default_params = {
        #     "min_samples_leaf": 50,
        #     "max_features": "sqrt",
        #     "min_samples_split": 400,
        #     "max_depth": 10,
        #     "subsample": 0.8,
        #     "learning_rate": 0.1,
        #     "random_state": 42,
        #
        #     "n_estimators": 40
        # }

        self.use_classifier = False
        # if self.use_classifier:
        #     regressor_default_params = {
        #         'learning_rate': 0.1,
        #         'min_samples_split': 200,
        #         'min_samples_leaf': 50,
        #         'max_depth': 13,
        #         'max_features': 11,
        #         'subsample': 0.8,
        #         'random_state': 8,
        #         'n_estimators': 220
        #     }

        if regressor_params is None:
            self.regressor = GradientBoostingRegressor(**regressor_default_params)
        else:
            self.regressor = GradientBoostingRegressor(
                            **{**regressor_default_params, **regressor_params})

        classifier_params = {
            'learning_rate': 0.1,
            'min_samples_split': 800,
            'min_samples_leaf': 60,
            'max_depth': 11,
            'max_features': 19,
            'subsample': 0.8,
            'random_state': 10,
            'n_estimators': 100
        }

        self.classifier = GradientBoostingClassifier(**classifier_params)

    def export_to_file(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)

    def get_fraction(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        checkout_since_booking = X["checkin_since_booking"] + X["no_of_nights"]
        true_fraction = labels["cancel_since_booking"] / checkout_since_booking
        true_fraction[true_fraction < 0] = self.no_cancel_val
        # true_fraction[true_fraction > 1] = 1
        return true_fraction

    @staticmethod
    def activation(x):
        return x

    @staticmethod
    def inverse_activation(x):
        return x

    def fit(self, X: np.ndarray, labels: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples
        """
        print("Fitting estimator...")
        X, labels = X.copy(), labels.copy()
        fraction = self.get_fraction(X, labels)

        # this removes 600 rows
        filter_ = fraction <= 1
        X = X[filter_]
        fraction = fraction[filter_]
        labels = labels[filter_]

        if self.use_classifier:
            only_cancelled = labels["is_cancelled"] == 1
            self.regressor.fit(X[only_cancelled], fraction[only_cancelled])
            # self.regressor.fit(X[only_cancelled], labels.cancel_since_booking[only_cancelled])
            # self.regressor.fit(X, fraction)
            self.classifier.fit(X, labels["is_cancelled"])
        else:
            self.regressor.fit(X, self.activation(fraction))

        print("Done fitting estimator.")

    def predict(self, X: np.ndarray, alpha, beta=0) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        """
        fraction_pred = self.inverse_activation(self.regressor.predict(X))
        fraction_pred[fraction_pred < 0] = 0
        fraction_pred[fraction_pred > 1] = 1

        checkout_since_booking_test = X["checkin_since_booking"] + X["no_of_nights"]
        day_pred = np.array(fraction_pred * checkout_since_booking_test)
        day_pred[fraction_pred > alpha] = -1

        # day_pred = self.regressor.predict(X)
        if self.use_classifier:
            is_cancelled_proba = self.classifier.predict_proba(X)
            day_pred[is_cancelled_proba[:, 0] < alpha] = -1

        return day_pred



