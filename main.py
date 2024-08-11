import data
import server
import client
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # Load data and set parameters
    (x_train, y_train), (x_test, y_test), (x_public, y_public) = data.load_mnist()
    client_count = 4
    batch_size = 40
    rounds = 10

    # Inititalize a client object with the public data and number of classes
    server = server.Server(x_public, y_public, client_count, 10)
    clients = []
    # split the data evenly
    length = len(x_train)//client_count
    # create x clients with unique datasets
    for i in range(client_count):
        train = x_train[i*length:(i+1)*length]
        test = y_train[i*length:(i+1)*length]
        cl = client.Client(train, test, batch_size, client_count)
        clients.append(cl)

    # access public data from server
    public_x, public_y = server.get_data()
    # ignore the commented code - used for testing other solutions
    clients_weight = [1/client_count for i in range(client_count)]
    # repeat round # of times
    for i in range(rounds):
        if i > 0:
            total = sum(clients_weight)
            for j in range(len(clients)):
                clients_weight[j] = clients[j].pick_weight/total
        # returns the list of the clients selected for training, which is half the clients
        indices = server.client_select(clients, client_count//2, clients_weight)
        # this loop can be altered to create multiple WL's for each client per round
        for k in range(1):
            # models = [[], [], []]
            # for each client, fit (with batches) and predict on public, return weight (calc error) and predictions
            for cl in indices:
                cl.fit(True)
                weight, preds = cl.get_weight(public_x, public_y)
                # self_pred = cl.self_predict()
                # cl.refactor_data(weight, self_pred)
                # models[0].append(cl.get_model())
                # models[1].append(weight)

                # add the weights and WL to the array in the server
                server.add_wl(cl.get_model(), weight)
                # resize the public data based on accuracy
                server.refactor_data(weight, preds)
                # public_x, publix_y = server.get_data()

                cl.reset_model()
            # for j in models[0]:
            #     models[2].append(accuracy_score(public_y, j.predict(public_x)))
            # index = models[2].index(max(models[2]))
            # server.add_wl(models[0][index], models[1][index])
            # server.refactor_data(models[1][index], models[0][index].predict(public_x))
            # public_x, publix_y = server.get_data()

        # test the adaboost model on the test data and print results
        pred = server.predict(x_test)
        print("Accuracy: " + str(accuracy_score(y_test, pred)))

    # results from pure decision tree
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    print(accuracy_score(y_test, tree.predict(x_test)))




