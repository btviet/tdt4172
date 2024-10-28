import pandas as pd
import numpy as np

train = pd.read_csv('final_mission_train.csv')
test = pd.read_csv('final_mission_test.csv')

print(train['nexus_rating'].value_counts())
