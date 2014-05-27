from rbfnnpy import *


def main():
    # Train model from specific file containing train data

    train_data = loadtxt('train.csv', delimiter=',')[range(10),:]
    train_target = loadtxt('target.csv', delimiter=',')[range(10),:]

    print 'Training network on', n, 'samples'
    start = time()
    rbf = Rbf(extra_neurons = k)

    rbf.train(train_data[:n, ], train_target[:n, ])
    now = time()
    
    print 'Time elapsed: {:.2f} (s)'.format(now - start)
    print rbf.summary
    print (mean(abs(model.predict(train_data) - train_target)))
   

if __name__ == "__main__":
    main()
