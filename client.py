from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math
import random


class Client:
    # assign initial values for class variables
    def __init__(self, x, y, batch_size, count):
        self.client_x = x
        self.client_y = y
        self.batch_size = batch_size
        self.round = 0
        self.acc = 0
        self.pick_weight = 1/count
        self.model = DecisionTreeClassifier()
        self.weight = 0
        self.data_weights = [1/(len(self.client_x)) for _ in range(len(self.client_x))]

    # if batches are used this will select the next batch in line for testing
    def batch_pick(self):
        start, end = self.batch_size * self.round, self.batch_size * (self.round + 1)
        x, y = self.client_x[start:end], self.client_y[start:end]
        self.round += 1
        if (self.round * self.batch_size > len(self.client_x)):
            self.round = 0
        return x, y

    # function for fitting the weak learner classifiers. If batch is true the fit is only done on a subset of the data.
    def fit(self, batch=False):
        x, y = self.client_x, self.client_y
        if batch:
            if self.round * self.batch_size > len(self.client_x):
                raise Exception("Insufficient data length for batch_size and round_count")
            x, y = self.batch_pick()
        self.model.fit(x, y)

    # this works the same way as the server's refactor_data function
    # Currently not being used but I considered using this function to do adaboost on local datasets
    def refactor_data(self, current_say, pred):
        length = len(self.client_x)
        total_weight = 0
        for i in range(length):
            if self.client_y[i] == pred[i]:
                self.data_weights[i] *= math.pow(math.e, current_say)
            else:
                self.data_weights[i] *= math.pow(math.e, -current_say)
            total_weight += self.data_weights[i]

        for i in range(length):
            self.data_weights[i] /= total_weight
        new_dataset = [[], []]
        for i in range(length):
            index = random.random()
            for j in range(length-1, 0, -1):
                if index > self.data_weights[j] or index < 0 or j == 0:
                    new_dataset[0].append(self.client_x[j])
                    new_dataset[1].append(self.client_y[j])
                    break
                index -= self.data_weights[j]
        self.client_x, self.client_y = new_dataset[0], new_dataset[1]

    # We need to create a new classifier after every fit or the previous fit will alter our results (I think)
    def reset_model(self):
        self.model = DecisionTreeClassifier()

    def predict(self, x):
        return self.model.predict(x)

    def self_predict(self):
        return self.model.predict(self.client_x)

    def get_labels(self):
        return self.client_y

    # calculates the weight of the current classifier
    def get_weight(self, data_x, data_y):
        predictions = self.model.predict(data_x)
        self.acc = accuracy_score(data_y, predictions)
        error = 1-self.acc
        self.weight = 0.5 * math.log((1-error)/error)
        self.pick_weight = self.acc
        return self.weight, predictions

    def get_model(self):
        return self.model

