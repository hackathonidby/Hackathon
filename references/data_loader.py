import os
import pickle

import numpy as np
import pandas as pd
import datetime as dt
from dictionaries import exchange_rate_dict, country_name_to_code_dict, gpd_dict, code_to_region_dict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
# sys.path.append("../")

cols_special_requests = [
    "request_nonesmoke",
    "request_latecheckin",
    "request_highfloor",
    "request_largebed",
    "request_twinbeds",
    "request_airport",
    "request_earlycheckin",
]

# columns with NaN value that needs to be 0
cols_nan_to_zero = cols_special_requests

numerical_features = [
    'original_selling_amount',
    'selling_amount_usd'
]

# the features of the preprocessed data
selected_features = [
    "hotel_star_rating",
    "guest_is_not_the_customer",

    "no_of_adults",
    "no_of_children",
    "no_of_extra_bed",
    "no_of_room",
    # "no_of_extra_bed",    # low correlation

    "original_selling_amount",
    "selling_amount_usd",

    "is_user_logged_in",
    "is_first_booking",

    "request_nonesmoke",
    # "request_latecheckin",  # low correlation
    # "request_highfloor",    # low correlation
    "request_largebed",
    # "request_twinbeds",     # low correlation
    # "request_airport",      # low correlation
    # "request_earlycheckin", # low correlation
    "total_special_requests",

    "no_of_nights",
    "checkin_since_booking",

    "noshow_penalty",
    "first_penalty",
    "first_ndays",
    "second_penalty",
    "second_ndays",

    "is_foreign_quest",
    "guest_country_wealth",

    "country_counter",
    "country_cancel_rate",

    "cancel_rate_per_customer",
    "booking_counter_per_customer",
    "cancel_rate_per_hotel",
    "booking_counter_per_hotel",

    # categorical
    "original_payment_type",
    "charge_option",
    "accommadation_type_name",
    "booking_month",
    "booking_dayofweek",

    'hotel_country_code',
    # 'customer_nationality',
    'guest_nationality_country',
    # 'origin_country_code',
    # 'language',
    'original_payment_method',
]


drop_outliers_features = [
    "original_selling_amount",
    "selling_amount_usd",
    # "total_special_requests",
    "no_of_nights",
    "checkin_since_booking",
    "noshow_penalty",
    "first_penalty",
    "first_ndays",
    "second_penalty",
    "second_ndays",
]


categorical_features = [
    "original_payment_type",
    "charge_option",
    "accommadation_type_name",
    "booking_month",
    "booking_dayofweek",

    'hotel_country_code',
    # 'customer_nationality',   # useless
    'guest_nationality_country',
    # 'origin_country_code',    # useless
    # 'language',               # useless
    'original_payment_method', # counter encode
]

# the target columns is cancelled_in_N where cancelled_in_N=1 iff the booking was
# cancelled within N days since the booking date. otherwise, cancelled_in_N=0
target_min_N = 7
target_max_N = 46
TARGETS = ["cancelled_in_" + str(n) for n in range(target_min_N, target_max_N + 1, 1)]
TARGETS.append("cancel_since_booking")
TARGETS.append("is_cancelled")
TARGETS.append("booking_date")
TARGETS.append("cancellation_fraction")


class Preprocessor:
    CACHE_FILENAME = '../pickles/cached_preprocessor.pkl'

    def __init__(self, use_cache=True):
        """
        Initialize the preprocessor
        """
        self.version = 20
        self.use_cache = use_cache

        self.onehot_encoder = None
        self.scaler = None
        self.payment_method_counters = None

        self.customer_cancel_rate_dict = None        # customer_counter for each customer id
        self.customer_counter_dict = None    # number of cancellations for each customer id

        self.hotel_cancel_rate_dict = None
        self.hotel_counter_dict = None

        self.country_cancel_counters = None
        self.country_counters = None

        self.is_train = True
        self.X = None
        self.labels = None

        if use_cache and os.path.exists(self.CACHE_FILENAME):
            with open(self.CACHE_FILENAME, 'rb') as fid:
                cached_pp = pickle.load(fid)
                if cached_pp['version'] == self.version:
                    print("Preprocessor: using cached data")
                    self.__dict__.update(cached_pp)
                    self.is_train = True
                    return
        print("Preprocessor: initializing new instance")

    @staticmethod
    def country_name_to_code(country_name):
        """
        Convert country name to country code
        """
        if country_name not in country_name_to_code_dict:
            raise ValueError("Unknown country name :" + country_name)
        return country_name_to_code_dict[country_name]

    @staticmethod
    def currency_convert_to_usd(value, currency):
        """
        Receives a money amount and a currency and converts it to USD
        :param value: The amount of money to convert
        :param currency: The original currency
        :return: The worth of this money in USD
        """
        if currency not in exchange_rate_dict:
            raise ValueError("Unknown currency type:" + currency)
        return value / exchange_rate_dict[currency]

    @staticmethod
    def penalty_to_usd(penalty, no_of_nights, selling_amount_usd):
        """
        Convert string of penalty ('100P', '2N', etc) to actual US dollars
        """
        penalty_type = penalty[-1]  # P or N
        penalty_amount = int(penalty[:-1])
        if penalty_type == 'P':
            return selling_amount_usd * (penalty_amount / 100)
        elif penalty_type == 'N':
            frac = min(1, penalty_amount / no_of_nights)
            return frac * selling_amount_usd
        else:
            raise ValueError("Unexpected penalty value: " + str(penalty))

    @staticmethod
    def calc_fraction(partial_days, total_days) -> np.ndarray:
        true_fraction = partial_days / total_days
        true_fraction[true_fraction < 0] = 1
        true_fraction[true_fraction > 1] = 1
        return true_fraction

    @staticmethod
    def process_labels(X):
        X["cancel_since_booking"] = X["cancellation_datetime"] - X["booking_datetime"]
        X.loc[X.cancel_since_booking < 0, "cancel_since_booking"] = -1
        X["is_cancelled"] = 0
        X.loc[X.cancel_since_booking >= 0, "is_cancelled"] = 1

        # add target columns: cancelled_in_N is 1 iff booking was cancelled within N days from the booking date
        for n in range(target_min_N, target_max_N + 1, 1):
            X["cancelled_in_" + str(n)] = 0
            X.loc[(X.cancel_since_booking >= 0) & (X.cancel_since_booking <= n), "cancelled_in_" + str(n)] = 1

        X["cancellation_fraction"] = Preprocessor.calc_fraction(
            partial_days=X["cancel_since_booking"],
            total_days=X["checkin_since_booking"] + X["no_of_nights"]
        )
        return X

    @staticmethod
    def process_boolean_features(X):
        """
        convert "is_user_logged_in" and "is_first_booking" from "True"\"False" to 1/0
        """
        X["is_user_logged_in"] = X["is_user_logged_in"].map({True: 1, False: 0})
        X["is_first_booking"] = X["is_first_booking"].map({True: 1, False: 0})
        return X

    @staticmethod
    def process_countries(X):
        """
        Process country related features:
        1. guest_nationality_country - convert from name to code
        2. is_foreign_quest -  is 1 iff guest_nationality_country equals hotel_country_code
        3. guest_country_wealth - is the wealth score of guest_nationality_country
        """
        # TODO: take data from here: https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv
        X["guest_nationality_country"] = X["guest_nationality_country_name"]

        filter_ = X.guest_nationality_country == "UNKNOWN"
        X.loc[filter_, "guest_nationality_country"] = X.customer_nationality[filter_]

        filter_ = X.guest_nationality_country.isnull()
        X.loc[filter_, "guest_nationality_country"] = X.customer_nationality[filter_]

        if X.guest_nationality_country.isnull().values.any():
            print("Unable to convert some countries:", file=sys.stderr)
            print(X.guest_nationality_country.isnull(), file=sys.stderr)

        X["guest_nationality_country"] = X["guest_nationality_country"].map(country_name_to_code_dict)

        X["is_foreign_quest"] = 0
        X.loc[X.guest_nationality_country == X.hotel_country_code, "is_foreign_quest"] = 1

        X["guest_country_wealth"] = X["guest_nationality_country"].map(gpd_dict)
        X.loc[X.guest_country_wealth == 0, "guest_country_wealth"] = gpd_dict["WORLD"]
        X.loc[X.guest_country_wealth.isnull(), "guest_country_wealth"] = gpd_dict["WORLD"]

        # convert country codes to region names
        X["guest_nationality_country"] = X["guest_nationality_country"].map(code_to_region_dict)
        X["hotel_country_code"] = X["hotel_country_code"].map(code_to_region_dict)

        return X

    def process_dates(self, X):
        """
        Process the dates in the dataframe X into int values
        """
        X["booking_month"] = pd.DatetimeIndex(X['booking_datetime']).month
        X["booking_dayofweek"] = pd.DatetimeIndex(X['booking_datetime']).dayofweek
        X["booking_date"] = pd.to_datetime(X["booking_datetime"])

        # TODO: remove checkin_month
        # X["checkin_month"] = pd.DatetimeIndex(X['checkin_date']).month
        # selected_features.append("checkin_month")

        # convert "date" columns from strings to ints
        # column names in the data that has date type
        date_type_cols = [
            "booking_datetime",
            "checkin_date",
            "checkout_date",
            "hotel_live_date",
        ]
        if self.is_train:
            date_type_cols.append("cancellation_datetime")

        for col in date_type_cols:
            X[col] = pd.to_datetime(X[col])
            X[col] = X[col].map(dt.datetime.toordinal)

        X["no_of_nights"] = X["checkout_date"] - X["checkin_date"]
        X.loc[X.no_of_nights < 0, "no_of_nights"] = 0

        X["checkin_since_booking"] = X["checkin_date"] - X["booking_datetime"]
        X.loc[X.checkin_since_booking < 0, "checkin_since_booking"] = 0
        return X

    def process_categorical_features(self, X):
        """
        Process categorical features
        """
        pass

    def process_cancellation_policy(self, X):
        """
        Process cancellation policy into the following features:
        1. noshow_penalty: the penalty in USD for no show
        2. first_penalty: the first penalty in USD
        3. first_ndays: the number of days after booking that the first penalty starts
        4. second_penalty:
        5. second_ndays:
        """

        days_matches = X["cancellation_policy_code"].str.extractall("([\\d]+)(D)")[0].unstack()
        penalty_matches = X["cancellation_policy_code"].str.extractall("(D)([\\d]+[NP])")[1].unstack()
        noshow_matches = X["cancellation_policy_code"].str.extract("(_)([\\d]+[NP])")[1]

        X["noshow_penalty"] = noshow_matches
        X["first_penalty"] = penalty_matches[0]
        X["first_ndays"] = days_matches[0]
        X["second_penalty"] = penalty_matches[1]
        X["second_ndays"] = days_matches[1]

        # if noshow_penalty is Nan assign "100P" to it
        X.loc[X.noshow_penalty.isnull(), "noshow_penalty"] = "100P"

        # if first_penalty is NaN assign noshow_penalty to it
        X.loc[X.first_penalty.isnull(), "first_penalty"] = X["noshow_penalty"]

        # if first_ndays is NaN assign 365 to it
        X.loc[X.first_ndays.isnull(), "first_ndays"] = 365

        # if second_penalty is NaN assign first_penalty to it
        X.loc[X.second_penalty.isnull(), "second_penalty"] = X["first_penalty"]

        # if second_ndays is NaN assign first_ndays to it
        X.loc[X.second_ndays.isnull(), "second_ndays"] = X["first_ndays"]

        X["noshow_penalty"] = X[['noshow_penalty', 'no_of_nights', 'selling_amount_usd']].apply(
            lambda x: self.penalty_to_usd(*x), axis=1)
        X["first_penalty"] = X[['first_penalty', 'no_of_nights', 'selling_amount_usd']].apply(
            lambda x: self.penalty_to_usd(*x), axis=1)
        X["second_penalty"] = X[['second_penalty', 'no_of_nights', 'selling_amount_usd']].apply(
            lambda x: self.penalty_to_usd(*x), axis=1)

        for col in ["first_ndays", "second_ndays"]:
            X[col] = X[col].astype(int)
            X.loc[X[col] == 999, col] = 365

            X[col] = X["checkin_since_booking"] - X[col]
            X.loc[X[col] < 0, col] = 0

            total_days = X["checkin_since_booking"] + X["no_of_nights"]
            fraction = X[col] / total_days
            fraction[fraction < 0] = 0
            fraction[fraction > 1] = 1
            X[col] = fraction

        return X

    def filter_rows(self, X):
        """
        filter outliners in the dataframe X, get called only on train data
        """
        if not self.is_train:
            return X

        relevant_range = {
            'original_selling_amount': [4, 3000],
            'selling_amount_usd': [0.001, 5000000],
            'no_of_nights': [1, 30],
            'checkin_since_booking': [7, 365],
        }

        for col, val in relevant_range.items():
            X = X.loc[(X[col] >= val[0]) & (X[col] <= val[1]), :]
        X.reset_index(drop=True, inplace=True)
        return X

    def make_dummies(self, X):
        """
        Convert categorical features to dummies
        """
        enc = self.onehot_encoder
        if enc is None:
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(X[categorical_features])

        transformed = enc.transform(X[categorical_features]).toarray()
        f = pd.DataFrame(transformed, columns=enc.get_feature_names_out())
        X = pd.concat([X, f], axis=1)
        X.drop(categorical_features, axis=1, inplace=True)

        self.onehot_encoder = enc
        return X

    def normalize_features(self, X):
        """
        Normalize features
        """
        numerical_features = [
            'original_selling_amount',
            'selling_amount_usd',
            "hotel_star_rating",
            "no_of_adults",
            "no_of_children",
            "no_of_extra_bed",
            "no_of_room",
            "total_special_requests",
            # "no_of_nights",
            # "checkin_since_booking",
            "guest_country_wealth",
            "country_counter",
            "country_cancel_rate",
        ]

        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X[numerical_features])

        X[numerical_features] = self.scaler.transform(X[numerical_features])
        return X

    def process_counters(self, X):
        X = self.process_customer_counters(X)
        X = self.process_hotel_counters(X)
        X = self.process_country_counters(X)
        return X

    def process_customer_counters(self, X):
        """
        Adds the columns:
        1. cancel_rate_per_customer - number of bookings per customer
        2. booking_counter_per_customer - rate of cancellation per customer
        """
        id_col = "h_customer_id"
        ids = X[id_col].unique()
        zeros = np.zeros(ids.size)
        rate_dict = dict(zip(ids, zeros))
        counter_dict = dict(zip(ids, zeros))

        if self.is_train:
            self.customer_cancel_rate_dict = rate_dict
            self.customer_counter_dict = counter_dict
            X.sort_values(by=["booking_date"])
        else:
            self.customer_cancel_rate_dict.update(rate_dict)
            self.customer_counter_dict.update(counter_dict)

        self.make_counters(X, self.customer_cancel_rate_dict, self.customer_counter_dict,
                           per="customer", id_col=id_col)
        return X

    def process_hotel_counters(self, X):
        """
        Adds the columns:
        1. cancel_rate_per_hotel - number of bookings per hotel
        2. booking_counter_per_hotel - rate of cancellation per hotel
        """
        id_col = "hotel_id"
        ids = X[id_col].unique()
        zeros = np.zeros(ids.size)
        rate_dict = dict(zip(ids, zeros))
        counter_dict = dict(zip(ids, zeros))

        if self.is_train:
            self.hotel_cancel_rate_dict = rate_dict
            self.hotel_counter_dict = counter_dict
            X.sort_values(by=["booking_date"])
        else:
            self.hotel_cancel_rate_dict.update(rate_dict)
            self.hotel_counter_dict.update(counter_dict)

        self.make_counters(X, self.hotel_cancel_rate_dict, self.hotel_counter_dict,
                           per="hotel", id_col=id_col)
        return X

    def process_country_counters(self, X):
        """
        Adds the columns:
        1. country_counters - number of bookings per country
        2. country_cancel_rate - rate of cancellation per country
        """
        feature_to_count = "guest_nationality_country_name"
        if self.country_counters is None:
            self.country_counters = X.value_counts(feature_to_count).to_dict()
            self.country_cancel_counters = X[X.is_cancelled == 1].value_counts(feature_to_count).to_dict()

        X["country_counter"] = X[feature_to_count].map(self.country_counters)
        cancel_counter = X[feature_to_count].map(self.country_cancel_counters)
        X["country_cancel_rate"] = cancel_counter / X["country_counter"]

        X.loc[X.country_counter.isnull(), "country_counter"] = 0
        X.loc[X.country_cancel_rate.isnull(), "country_cancel_rate"] = 0
        return X

    def make_counters(self, X, rate_dict, counter_dict, per, id_col):
        X["cancel_rate_per_"+per] = 0
        X["booking_counter_per_"+per] = 0

        for i, row in X.iterrows():
            id_ = row[id_col]
            X.loc[i, "cancel_rate_per_"+per] = rate_dict[id_]
            X.loc[i, "booking_counter_per_"+per] = counter_dict[id_]

            if self.is_train:
                is_cancel = row["is_cancelled"]
                counter = counter_dict[id_]
                rate_dict[id_] = (rate_dict[id_] * counter + is_cancel) / (counter + 1)
                counter_dict[id_] += 1

    def load_data(self, filename, convert_dummies=True):
        """
        Preprocess the data and return the dataFrame of it
        :param convert_dummies: if True, convert categorical features to dummies
        :param filename: the path of the data
        """
        if self.is_train and self.X is not None:
            # return cached data
            self.is_train = False
            return self.X, self.labels

        X = pd.read_csv(filename)

        # convert values in cols_nan_to_zero  from NaN to 0
        for col in cols_nan_to_zero:
            X.loc[X[col].isnull(), col] = 0



        # total_special_requests is the sum of all the cols of "request_..."
        X['total_special_requests'] = 0
        for col in cols_special_requests:
            X['total_special_requests'] += X[col]

        X = self.process_dates(X)

        # convert "is_user_logged_in" and "is_first_booking" from "True"\"False" to 1/0
        X = self.process_boolean_features(X)

        # convert selling amount to usd
        convert_currency_vector = np.vectorize(self.currency_convert_to_usd)
        X["selling_amount_usd"] = convert_currency_vector(X["original_selling_amount"], X["original_payment_currency"])

        # X = self.filter_rows(X)

        # ===== Proccess Categorical Features =======
        X = self.process_countries(X)

        # == charge_option ==
        # value:  Pay Later  |  Pay Now  |  Pay at Check-in
        # count:     14454   |   44170   |       35
        X.loc[X.charge_option == "Pay at Check-in", "charge_option"] = "Pay Later"

        # == original_payment_type ==
        # value:  Credit Card  |  Gift Card  |  Gift Card
        # count:     57643     |     230     |    786

        most_common_accommadation_type = [
            'Hotel',
            'Resort',
            'Serviced Apartment',
            'Guest House / Bed & Breakfast',
            'Hostel',
            'Apartment',
        ]
        not_common_rows = ~X.accommadation_type_name.isin(most_common_accommadation_type)
        X.loc[not_common_rows, "accommadation_type_name"] = "other_accommadation"

        X.loc[X.hotel_star_rating == -1, "hotel_star_rating"] = 0#3

        # keep only selected cols
        if self.is_train:
            # clean
            X = self.process_labels(X)
            Y = X[TARGETS]
        else:
            Y = pd.DataFrame(X["booking_date"])

        X = self.process_counters(X)

        # normalize prices
        # X = self.normalize_features(X)

        X = self.process_cancellation_policy(X)

        X = X[selected_features]

        # convert categorical features to dummies
        if convert_dummies:
            # X = self.counter_encode(X)
            X = self.make_dummies(X)

        if self.is_train and self.use_cache and convert_dummies:
            self.X, self.labels = X, Y
            self.save_cache()

        self.is_train = False
        return X, Y

    def save_cache(self):
        with open(self.CACHE_FILENAME, 'wb') as fid:
            pickle.dump(self.__dict__, fid)

    def load_data_test(self, filename):
        """
        Preprocess the data and return the dataFrame of it
        :param convert_dummies: if True, convert categorical features to dummies
        :param filename: the path of the data
        """
        if self.is_train:
            raise Exception("You should load data for training first")
        return self.load_data(filename, convert_dummies=True)

    def counter_encode(self, X):
        if self.payment_method_counters is None:
            self.payment_method_counters = X.groupby(["original_payment_method"]).count().iloc[:, 0].to_dict()
        X["original_payment_method"] = X["original_payment_method"].map(self.payment_method_counters)
        return X


if __name__ == '__main__':
    np.random.seed(0)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 7)
    pd.set_option('display.width', 150)

    # Load data
    preprocessor = Preprocessor(use_cache=False)
    X, labels = preprocessor.load_data("agoda_cancellation_train.csv", convert_dummies=False)

    X_9, labels_9 = preprocessor.load_data(f"datasets/new_tests/week_9_test_data.csv", convert_dummies=False)
