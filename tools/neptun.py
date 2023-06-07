import neptune.new as neptune
import pandas as pd

run = neptune.init(project='your_username/your_project_name',
                   api_token='YOUR_API_TOKEN')

data = pd.read_csv("Agoda_training_data.csv")
params = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 0,
}
