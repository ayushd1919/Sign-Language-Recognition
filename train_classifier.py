import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert data and labels to NumPy arrays
data = data_dict['data']
labels = data_dict['labels']

# Check the lengths of the data entries
data = [d for d in data if len(d) == 42]  # Assuming you expect 21 landmarks (2 values each)
labels = [labels[i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == 42]

# Convert to NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

