import pandas as pd
import datetime as dt
import numpy as np

class Preprocess:
    RELEVANT_COLS = []

    @staticmethod
    def preprocess_data(data):
        data = Preprocess.process_dates(data)
        # data = Preprocess._preprocess_categoricals(data)
        # data = Preprocess._preprocess_numericals(data)

        # Step 8: Convert "charge_option" string values to numbers
        data['charge_option'] = data['charge_option'].map({'pay now': 1, 'pay_later': 2, 'pay_checkin': 3})

        # Step 9: Convert "is_first_booking" column to 0 and 1 columns instead of False and True
        data['is_first_booking'] = data['is_first_booking'].astype(int)

        # Step 10: Create new columns from cancellation_datetime
        # data['does_canceled'] = data['cancellation_datetime'].isnull().astype(int)
        dates_col = data.filter(like='date').columns
        columns_to_drop = ['request_nonesmoke', 'request_highfloor', 'request_airport', 'request_largebed',
                       'request_twinbeds', 'hotel_id', 'hotel_city_code', 'hotel_brand_code', 'hotel_chain_code']

        # Drop the specified columns
        data = data.drop(columns=columns_to_drop)
        data = data.drop(columns=dates_col)
        Preprocess._remove_if_string(data)
        Preprocess.preprocess_cancellation_policy_code(data)
        data = data.select_dtypes(include=[np.number])
        data = data.dropna()
        return data
    
    @staticmethod
    def convert_to_percentage(x, duration):
        """
        If the penalty is a night based, convert it to a percentage based
        """
        if "N" in x:
            return int(x.replace("N",""))/duration*100
        elif "P" in x:
            return float(x.replace("P",""))
        else:
            return x
    @staticmethod
    def parse_policy(x):
        """
        Parse the policy string to a list of conditions and penalties
        """
        conditions = []
        penalties = []

        for i in x.split('_'):
            c, p = i.split('D') if 'D' in i else ('0', i)
            conditions.append(int(c))
            penalties.append(p)
        
        return conditions, penalties
    @staticmethod
    def preprocess_cancellation_policy_code(df):
        """
        Preprocess the DataFrame df
        """
        df['conditions'], df['penalties'] = zip(*df['cancellation_policy_code'].map(Preprocess.parse_policy))
        df['penalties'] = df.apply(lambda row: [Preprocess.convert_to_percentage(x, row['vacation_duration']) for x in row['penalties']], axis=1)
        df['first_condition'] = df['conditions'].apply(lambda x: x[0] if len(x) > 1 else 'NONE')
        df['second_condition'] = df['conditions'].apply(lambda x: x[1] if len(x) > 1 else 'NONE')
        df['first_penalty'] = df['penalties'].apply(lambda x: x[0] if len(x) > 1 else 'NONE')
        df['second_penalty'] = df['penalties'].apply(lambda x: x[1] if len(x) > 1 else 'NONE')
        df['no_show'] = df['penalties'].apply(lambda x: x[-1])
        df.drop(columns=['conditions', 'penalties'], inplace=True)

        return df
    
    @staticmethod
    def _remove_outliers(data):
        pass
    @staticmethod
    def _remove_if_string(data):
        # removes columns that are strings:
        for col in data.columns:
            if data[col].dtype == type(object):
                data = data.drop(col, axis=1)
        data = data.select_dtypes(exclude=['object'])
        return data

    @staticmethod
    def _preprocess_categoricals(data):
        pass
        # data['accommodation_type_name'] = data['accommodation_type_name'].get_dummies()

    @staticmethod
    def _preprocess_numericals(data):
        pass
    @staticmethod
    def process_dates(X):
            """
            Process the dates in the dataframe X into int values
            """
            # X["booking_month"] = pd.DatetimeIndex(X['booking_datetime']).month
            # X["booking_dayofweek"] = pd.DatetimeIndex(X['booking_datetime']).dayofweek
            # X["booking_date"] = pd.to_datetime(X["booking_datetime"])

            # TODO: remove checkin_month
            # X["checkin_month"] = pd.DatetimeIndex(X['checkin_date']).month
            # selected_features.append("checkin_month")

            # convert "date" columns from strings to ints
            # column names in the data that has date type

            # for date_col in X.filter(like='date').columns:
            #     X[date_col] = pd.to_datetime(X[date_col])
            #     X[date_col] = X[date_col].map(dt.datetime.toordinal)

            
            for date_col in X.filter(like='date').columns:
                    X[date_col] = pd.to_datetime(X[date_col]).dt.date
            X['vacation_duration'] = (X['checkout_date'] - X['checkin_date']).dt.days
            X.loc[X['vacation_duration'] < 0, 'vacation_duration'] = pd.NaT

            X['how_long_booking_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days
            X.loc[X['how_long_booking_before_checkin'] < 0, 'how_long_booking_before_checkin'] = pd.NaT

            X['how_old_hotel'] = (X['booking_datetime'] - X['hotel_live_date']).dt.days

            # Step 6: Ensure hotel_star_rating values are between 1 and 5, imputing the median for illegal values
            median_star_rating = X['hotel_star_rating'].median()
            X.loc[
                (X['hotel_star_rating'] < 1) | (X['hotel_star_rating'] > 5), 'hotel_star_rating'] = median_star_rating

            # X['days_before_checkin_date'] = (X['checkin_date'] - X['cancellation_datetime']).dt.days
            # X = X[X['cancellation_datetime'] < X['checkout_date']]

            return X