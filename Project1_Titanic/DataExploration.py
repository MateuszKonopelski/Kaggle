# Import neccessary packages
import pandas as pd

# Import CSV into data frame
train = 'data\\train.csv'
test = 'data\\test.csv'

train_data = pd.read_csv(train, nrows=10)
test_data = pd.read_csv(train, nrows=10)

