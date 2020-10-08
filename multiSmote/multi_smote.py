import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class multiSmote():
    """
    Multi Smote, is an extension of the Synthetic Minority Over Sampling Technique that
    supports multi label data. It get the samples of the minority class, sample that belongs
    in only one class are considered in order to generate data without affecting the other classes
    at the data augmentation process keeping the number of samples fixed.
    """

    def __init__(self):
        """
        Initializing multi smote algorithm
        """
        self.neighbors = 5
        self.class_bins = [] # contains the sum of the class instances


    def get_classes(self, y) -> int:
        """
        Get the total number of classes.

        Args:
            y : The labels list of the data.

        Returns:
            int: Number of total classes
        """

        return int(y.shape[1])



    def get_sum_classes(self, y)->list:
        """
        Get the number of samples per class

        Args:
            y : Labels of the data

        Returns:
            list: [description]
        """

        if isinstance(y, pd.DataFrame):
            return pd.DataFrame(y).sum().tolist()
        if isinstance(y, np.ndarray):
            return y.sum(axis=0)


    def get_majority_index(self)->int:
        """
        Get the index of the majority class

        Returns:
            int: The index of the majority class
        """

        return np.argmax(self.class_bins)


    def get_minority_index(self)->int:
        """
        Get the index of the minority class

        Returns:
            int: The index of the minrity class
        """

        return np.argmin(self.class_bins)


    def get_majority_class(self)->int:
        """
        Get the number of samples from the majority class

        Returns:
            int: The number of samples
        """

        return np.max(self.class_bins)

    def get_minority_class(self)->int:
        """
        Get the number of samples from the minority class

        Returns:
            int: The number of samples
        """

        return np.min(self.class_bins)

    def get_minority_samples(self, X, y)->tuple:
        """
        Get the samples and labels from the minority class

        Args:
            X : The Data
            y : The Labels

        Returns:
            tuple: [The minority's class samples, The minority class labels]
        """

        assert X.shape[0] == y.shape[0], "Samples and labels are not in the same length"
        assert type(X) == type(y), "Data types of X and y does not match"

        index = int(self.get_minority_index()) # index of the minority class


        if isinstance(X, pd.DataFrame):
            "Dataframe support"
            x_sub = []
            y_sub = []
            for row in range(X.shape[0]):
                if y.iloc[row].sum() == 1 and y.iloc[row, index] == 1:
                    x_sub.append(X.iloc[row])
                    y_sub.append(y.iloc[row])
            return pd.DataFrame(data=x_sub, index=None), pd.DataFrame(data=y_sub, index=None)

        elif isinstance(X, np.ndarray):
            "Numpy support"
            x_sub = []
            y_sub = []
            for row in range(X.shape[0]):
                if y[row].sum() == 1 and y[row][index] == 1:
                    x_sub.append(X[row])
                    y_sub.append(y[row])
            return np.array(x_sub),np.array(y_sub)

    def nearest_neighbour(self, X)->list:
        """
        Calculate the nearest Neighbors for the data

        Args:
            X : The samples from the minority class

        Returns:
            list: List of the nearest neighbors
        """

        nbs = NearestNeighbors(n_neighbors=self.neighbors, metric='euclidean', algorithm='kd_tree').fit(X)
        _, indices = nbs.kneighbors(X)
        return indices

    def resample(self, X, y)->list:
        """
        The function that produces synthetic data from the representative samples of the minority class

        Args:
            X : Samples
            y : Labels

        Returns:
            list: [synthetic samples, synthetic's labels]
        """
        self.class_bins = self.get_sum_classes(y) # Update the class bins.
        x_sub, y_sub = self.get_minority_samples(X, y) # Get the minority samples

        if len(x_sub) < self.neighbors and len(x_sub) > 1:
            print('Number of Minority samples are less than the number of nearest neighbors,'
                  ' trying to resolve the conflict by decreasing the number of the neighbors')
            print("New k={0}".format(len(x_sub)))
            self.neighbors = len(x_sub)

        if len(x_sub)<=1:
            print('The number of the unique samples from the minority class is small,'
                  ' cannot find neighbors for this minority class.\n'
                  'Aborting for class {}'.format(self.get_minority_index()))
            return None, None

        indices = self.nearest_neighbour(x_sub)

        # num_samples: the number of synthetic samples
        num_samples = int(self.get_majority_class() - self.get_minority_class())

        gen_x = [] # Generated sampled
        gen_y = [] # labels for generated samples

        for _ in range(num_samples):
            nn = random.randint(0, len(x_sub) - 1)
            # Random number from the neighbor's matrix
            neighbour = random.choice(indices[nn, 1:])
            # A random neighbor from the neighbour's matrix.
            ratio = random.random()
            if isinstance(x_sub, pd.DataFrame):
                "pandas support"
                gap = x_sub.iloc[nn, :] - x_sub.iloc[neighbour, :]
                generated = np.array(x_sub.iloc[nn, :] + ratio * gap)
                gen_x.append(generated)
                gen_y.append(y_sub.iloc[0])

            elif isinstance(y_sub, np.ndarray):
                "numpy support"
                gap = x_sub[nn, :] - x_sub[neighbour, :]
                generated = np.array(x_sub[nn, :] + ratio * gap)
                gen_x.append(generated)
                gen_y.append(y_sub[0])


        if isinstance(x_sub, pd.DataFrame):
            return pd.DataFrame(data=gen_x, index=None), pd.DataFrame(data=gen_y, index=None)
        elif isinstance(y_sub, np.ndarray):
            return np.array(gen_x), np.array(gen_y)


    def multi_smote(self, X, y)->list:
        """
        The main function of multi label smote. The function resample data for all the classes of the data.
        The returned data will be balanced, only if the representative data of each class is greater than one istance.


        Args:
            X : Data
            y : Labels

        Returns:
            list: [Resampled Data, Resampled Labels]
        """
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray):
            print("Not supported type of the data.\n"
                  " Aborting")
            return None

        classes = self.get_classes(y) # number ofclasses

        for _ in range(classes-1): #minus one, we exclude the majority class.
            x_new, y_new = self.resample(X, y)

            if x_new is not None and y_new is not None:
                if isinstance(X, pd.DataFrame):
                    "pandas support"
                    X = pd.concat([X, x_new],axis = 0)
                    y = pd.concat([y, y_new],axis =0)

                elif isinstance(y, np.ndarray):
                    "numpy support"
                    X = np.concatenate((X, x_new))
                    y = np.concatenate((y, y_new))
        del x_new, y_new, classes
        return X, y


    def __str__(self):
        """
        str Method in order to change the printed message of the multilabel smote object

        Returns:
            message object.
        """
        return "Multi Label Smote Object. Default k is {0}".format(self.neighbors)
