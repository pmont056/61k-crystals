from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models


#   Convert SMILES Strings into OneHotEncode 2D arrays
def encoded_smiles(raw_dataframe):
    df_62k = pd.read_json(raw_dataframe, orient='split')
    molecules = df_62k['canonical_smiles'][0:61489].to_list()

    #   Padding 0 at the end of the string so that the OHE list is the same length
    maxlen = len(max(molecules, key=len))
    molecules = [x + ('0' * (maxlen - len(x))) for x in molecules]

    for i, molecule in enumerate(molecules):
        molecules[i] = (list(molecule))
    x = np.array(molecules)

    enc = OneHotEncoder(handle_unknown='ignore')
    x1 = enc.fit(x)
    x1 = x1.transform(x).toarray()

    return x1


def pull_predicted_variable(name):
    df_62k = pd.read_json(raw_dataframe, orient='split')
    y_variable = df_62k[name][0:61489]
    return y_variable


#   Put file path here
raw_dataframe = '/Users/minhpham/Desktop/Machine Learning Project/m1507656/df_62k.json'

#   Put y-variable here
name = 'total_energy_pbe'

df = encoded_smiles(raw_dataframe)
y = pull_predicted_variable(name)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Reshape the images.
X_train = np.expand_dims(X_train, axis=4)
X_test = np.expand_dims(X_test, axis=4)

np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('y_train', y_train)
np.save('y_test', y_test)
num_filters = 8
filter_size = 3
pool_size = 2


print(model.summary())
# Build the model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=((61489, 283, 1), 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


# Train the model.
model.fit(
  X_train,
  to_categorical(y_train),
  epochs=5,
  validation_data=(X_train, to_categorical(y_test)),
)

# Save the model to disk.
model.save_weights('cnn.h5')

'''# Load the model from disk later using:
# model.load_weights('cnn.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]'''


