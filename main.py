import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bigdata.charts import Charts, plot_full_metrics
from bigdata.gui import get_categorical_columns, Gui
from bigdata.utils import train_and_evaluate_ffnn, train_and_evaluate_cnn

# Load the data
data = pd.read_csv("../datasets/Children_ASD.csv")
data = data.rename(columns={"ASD_traits": "ASD", 'A10_Autism_Spectrum_Quotient': 'A10'})
label_encoders = {}

# Convert Age_Years to Age_Mons
data[['Age_Mons']] = data[['Age_Years']] * 12

chart = Charts(data)
gui = Gui(chart)
gui.mainloop()


# Encode categorical features in Dataset 2
all_categorical_cols = ['ASD', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test']
categorical_cols = ['ASD'] + get_categorical_columns()
categorical_cols_difference = list(set(all_categorical_cols) - set(categorical_cols))

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le





# Features and target for Dataset 2
features = data.drop(columns=['ASD', 'Age_Years', 'Unnamed: 0'] + categorical_cols_difference)
target = data['ASD']


# Normalize numerical columns in Dataset 2
scaler = StandardScaler()
features[['Age_Mons']] = scaler.fit_transform(features[['Age_Mons']])


# Split Dataset 2 into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train on Dataset 2
print("\nTraining FFNN:")
ffnn_model, ffnn_history = train_and_evaluate_ffnn(x_train, y_train, x_test, y_test)
print(ffnn_history.history)

print('=========================')

print("\nTraining on CNN:")
cnn_model, cnn_history = train_and_evaluate_cnn(x_train, y_train, x_test, y_test)
print(cnn_history.history)


# Plot for FFNN
plot_full_metrics(ffnn_history, title="FFNN Training Metrics", model_name="FFNN")

# Plot for CNN
plot_full_metrics(cnn_history, title="CNN Training Metrics", model_name="CNN")

