import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
import pickle
import onnxruntime as rt


def etl(name):
    df = pd.read_csv(name)  # Load the data
    return df


def create_model_impl(df):
    # Testing a RandomForest model
    # The target variable is 'quality'.
    Y = df['quality']
    X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
    #X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides']]
    # Split the data into train and test data:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    # Build the model with the random forest regression algorithm:
    model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    model.fit(X_train, Y_train)
    return {'model':model, 'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}

curr_dir = os.path.dirname(os.path.realpath(__file__))
df = etl(curr_dir + '/winequality-red.csv')
model_state = create_model_impl(df)
model = model_state['model']
X_test = model_state['X_test']
Y_test = model_state['Y_test']

filename = 'models/random_forest_model.pkl'
pickle.dump(model, open(filename, 'wb'))
# Concept of dynamic axis applies here also.
# If I specify None as the number of rows, I can pass any number of rows to sess.run
# If I specify a fixed number, then sess.run will throw an assert if a different number of rows is passed
initial_types = [('float_input', FloatTensorType([None, 11]))]
onnx_model = onnxmltools.convert_sklearn(model, initial_types=initial_types)
onnxmltools.utils.save_model(onnx_model, 'models/random_forest_model.onnx')

onnx_model = onnxmltools.utils.load_model('models/random_forest_model.onnx')
# Reverse conversion from onnx to sklearn not possible?
sess = rt.InferenceSession('models/random_forest_model.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input = X_test.astype(np.float32).to_numpy()
# Input to sess.run is plain numpy array
start = time.time()
pred_onnx = sess.run([output_name], {input_name: input })[0]
print('Random Forest (ONNX) execution: {0}'.format(time.time() - start))
start = time.time()
pred_sklearn = model.predict(X_test)
print('Random Forest (sklearn) execution: {0}'.format(time.time() - start))

# assert pred_sklearn == pred_onnx
print('done')