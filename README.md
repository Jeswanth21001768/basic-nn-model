# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions. This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model
![image](https://github.com/Jeswanth21001768/basic-nn-model/assets/94155480/0731b851-25b3-41e2-8307-8f2189d78df0)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects, fit the model, and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Jeswanmth S
### Register Number: 212221230042
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('Exp01 DL').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'output':'float'})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('Exp01 DL').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head()


X = dataset1[['Input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
model = Sequential([
                    Dense(6,activation = 'relu'),
                    Dense(6,activation = 'relu'),
                    Dense(1)
                    ])
model.compile(optimizer = 'rmsprop', loss = 'mse')
model.fit(X_train1,y_train,epochs = 2000)

loss_df = pd.DataFrame(model.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

model.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

model.predict(X_n1_1)

```
## Dataset Information

### Include a screenshot of the dataset

![image](https://github.com/Jeswanth21001768/basic-nn-model/assets/94155480/ec35f4f5-7529-430b-9fb6-607721e3be05)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Jeswanth21001768/basic-nn-model/assets/94155480/18f3c385-af30-4ac4-afe6-7a96b234fa55)


### Test Data Root Mean Squared Error

![image](https://github.com/Jeswanth21001768/basic-nn-model/assets/94155480/05b2cfa2-4e6f-40a1-aacd-2a39068efc3c)


### New Sample Data Prediction

![image](https://github.com/Jeswanth21001768/basic-nn-model/assets/94155480/44b22e16-f878-43d1-9b87-c62236995df8)


## RESULT
Thus the neural network regression model for the given dataset has been developed successfully.


