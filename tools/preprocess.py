import pandas as pd
import datetime as dt


class Preprocess:
    RELEVANT_COLS = []

    @staticmethod
    def preprocess_data(data):
        data = Preprocess._preprocess_dates(data)
        data = Preprocess._preprocess_categoricals(data)
        data = Preprocess._preprocess_numericals(data)
        return data

    @staticmethod
    def _remove_outliers(data):
        pass
    @staticmethod
    def _remove_if_string(data):
        # removes columns that are strings:
        for col in data.columns:
            if data[col].dtype == object:
                data = data.drop(col, axis=1)
        return data

    @staticmethod
    def _preprocess_categoricals(data):
        pass

    @staticmethod
    def _preprocess_numericals(data):
        pass
    @staticmethod
    def process_dates(X):
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

            for col in date_type_cols:
                X[col] = pd.to_datetime(X[col])
                X[col] = X[col].map(dt.datetime.toordinal)

            X["no_of_nights"] = X["checkout_date"] - X["checkin_date"]
            X.loc[X.no_of_nights < 0, "no_of_nights"] = 0

            X["checkin_since_booking"] = X["checkin_date"] - X["booking_datetime"]
            X.loc[X.checkin_since_booking < 0, "checkin_since_booking"] = 0
            return X

    # booking_datetime-chekin_date
    # checking_in - checking_out = days
    # checking_date: day of week/weekend/summer/winter
    # hotel_live_date: how long the hotel has been in the system (if new when booked)?
    # hotel_star_rating: 0-5

    # hotel city code: how many hotels in the city?
    # hotel country name: how many hotels in the country?
    # hotel city code: turn to catrgorial