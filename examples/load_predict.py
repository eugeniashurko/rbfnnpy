from rbfnnpy import *


def main():
    # Load model parameters from files
    model = Rbf(from_files = {
                'mu'   : 'rbf-mu-212243-2400.hdf5',
                'sigma': 'rbf-sigma-212243-2400.hdf5',
                'w'    : 'rbf-w-212243-2400.hdf5',
            })
    print 'Loading data...'

    # 
    test_data = loadtxt('new_test.csv', delimiter=',')
    model.n = test_data.shape[0]
    model.predict(test_data)
    

if __name__ == "__main__":
    main()
