import math
import random
import numpy as np
from sklearn.metrics import accuracy_score

class Server:
    # assign parameters to be used in class functions
    # parameters for each object of a class are distinct to that object (think strings having unique lengths)
    def __init__(self, x, y, client_count, class_count):
        self.public_x = x
        self.data_weights = [1/len(self.public_x) for i in range(len(self.public_x))]
        self.public_y = y
        self.count = client_count
        self.weak_learners = []
        self.weak_learner_weights = []
        self.classes = class_count

    # Method that adjusts data point prevalence by weight
    def refactor_data(self, current_say, pred):
        length = len(self.public_x)
        total_weight = 0
        # for each value currently in dataset adjust weight with a formula depending on if the model correctly predicted it
        for i in range(length):
            if self.public_y[i] == pred[i]:
                self.data_weights[i] *= math.pow(math.e, current_say)
            else:
                self.data_weights[i] *= math.pow(math.e, -current_say)
            # determine value that total weights add to for normalization
            total_weight += self.data_weights[i]

        # divide each weight by the total_weight so total_weight of values adds to 1
        for i in range(length):
            self.data_weights[i] /= total_weight

        # the next loop repeats len(dataset) times.
        # randomly selects a value 0-1 and determines which value's weight range it is in
        # adds that value and its weight to the new dataset
        new_dataset = [[], []]
        for i in range(length):
            index = random.random()
            for j in range(length-1, 0, -1):
                if index > self.data_weights[j] or index < 0 or j == 0:
                    new_dataset[0].append(self.public_x[j])
                    new_dataset[1].append(self.public_y[j])
                    break
                index -= self.data_weights[j]
        # overwrite the dataset with the refactored one
        # self.public_x, self.public_y = new_dataset[0], new_dataset[1]

    # loops through the datapoints in x.
    # uses each weak learner to predict on the given datapoint, adding or subtracting from the predicted value based on weight
    # whichever classification value 0-9 has the highest value is the predicted choice
    # def predict(self, X):
    #     n_classes = len(self.weak_learners)
    #     n_samples = X.shape[0]
    #     predictions = np.zeros((n_samples, n_classes))
    #
    #     for i in range(len(self.weak_learners)):
    #         predictions[:, i] = self.weak_learner_weights[i] * self.weak_learners[i].predict(X)
    #
    #     return np.argmax(predictions, axis=1)
    #
    def predict(self, x):
        predictions = [0 for _ in range(len(x))]
        for i in range(len(x)):
            prediction = [0 for i in range(self.classes)]
            for j in range(len(self.weak_learners)):
                sample = x[i].reshape(1, -1)
                wl_pred = int(self.weak_learners[j].predict(sample)[0])
                prediction[wl_pred-1] += self.weak_learner_weights[j]
            predictions[i] = prediction.index(max(prediction))
        return predictions



    # adds the weak learner and its weight to the arrays for predictions
    def add_wl(self, wl, weight):
        self.weak_learners.append(wl)
        self.weak_learner_weights.append(weight)

    # currently it randomly selects count # of clients from clients_sampled
    def client_select(self, clients_sampled, count, clients_weights):
        clients = []
        for i in range(count):
            val = random.random()
            cum_weights = 0
            for j in range(len(clients_sampled)):
                if cum_weights > val:
                    clients.append(clients_sampled[j])
                    break
                cum_weights += clients_weights[j]
        print("Clients selected: " + str(clients))
        print("Client weights: " + str(clients_weights))
        return clients

    # return the dataset for use in other functions
    def get_data(self):
        return self.public_x, self.public_y

    # accuracy
    def get_accuracy(self, x, y_true):
        preds = self.predict(x)
        return accuracy_score(y_true, preds)




