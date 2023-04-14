# Forest Cover Type

Project is created by Adziareika Aliaksandr

[LinkedIn](https://www.linkedin.com/in/aadereiko/)  
[GitHub project](https://github.com/aadereiko/forest_covtype)

## Description

There are two main endpoints to get the data:

- Predict
- Get the example input

All the requests are provided with postman as well.  
All the models are saved in the **models** directory.

## Tasks

### Task 2

I decided to base Heuristics on Elevation. The range of values is divided into 7 about equal parts and depending on it the class is taken.

### Task 3

Machine Learning Models. I decided to use **Logistic Regression** and **SGD Classifier**. The first is pretty simple and basic, which became a reason. And SGD is chosen because the data set is quite huge and we solving the issues of classification.

### Task 4

Neural Networks. **RandomSearch** was used to adjust hyperparameters. And a few of **Dense**, **Dropout** layers have been added.

### Task 5

Evaluation. Some **Confusion matricies** were logged out, **accuracy metric**. **History charts** for NN are also shown.

### Task 6

REST API it touchable and playable :)

## Endpoints

### Predict

`POST`  
`/predict`

**Body:**
```buildoutcfg
{
    "model_name": "heuristics | nn | logistic | sgd",
    "input": [[...]]
}
```

### Get the example input
`GET`  
`/input/example`

**Response:**
```buildoutcfg
{
    "input": [[2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0]]
}
```