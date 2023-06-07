import pandas as pd
from plotly import graph_objects as go
import plotly.express as px

data = pd.read_csv(r"data\agoda_cancellation_train.csv")
print(data.head())
print(data.shape)
X = data.drop("is_canceled", axis=1)
y = data["is_canceled"]
# h_booking_id                          0
# booking_datetime                      0
# checkin_date                          0
# checkout_date                         0
# hotel_id                              0
# hotel_country_code                    4
# hotel_live_date                       0
# hotel_star_rating                     0
# accommadation_type_name               0
# charge_option                         0
# h_customer_id                         0
# customer_nationality                  0
# guest_is_not_the_customer             0
# guest_nationality_country_name        0
# no_of_adults                          0
# no_of_children                        0
# no_of_extra_bed                       0
# no_of_room                            0
# origin_country_code                   2
# language                              0
# original_selling_amount               0
# original_payment_method               0
# original_payment_type                 0
# original_payment_currency             0
# is_user_logged_in                     0
# cancellation_policy_code              0
# is_first_booking                      0
# request_nonesmoke                 25040
# request_latecheckin               25040
# request_highfloor                 25040
# request_largebed                  25040
# request_twinbeds                  25040
# request_airport                   25040
# request_earlycheckin              25040
# cancellation_datetime             42895
# hotel_area_code                       0
# hotel_brand_code                  43356
# hotel_chain_code                  42902
# hotel_city_code                       0
# dtype: int64
def features_with_big_correlation(data, threshold):
    for col in data.columns:
        if col != "is_canceled":
            corr = data[col].corr(data["is_canceled"])
            if abs(corr) > threshold:
                print(f"{col} - {corr}") 


def run_main():
    
if  __name__ == "__main__": 
     run_main()