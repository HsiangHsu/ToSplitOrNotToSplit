import numpy as np
import pandas as pd
from sklearn import svm

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util import *
'''

class BenefitSplit:
    
    #### Initialization ####
    def __init__(self, df, S_name, Y_name):
        """
        :df: dataset
        :S_name: sensitive attribute name
        :Y_name: label name
        """
        self.df = df
        self.S_name = S_name
        self.Y_name = Y_name
        # random seed
        self.rand_seed = 20202020
        # split dataset based upon sensitive attribute
        self.split_data()
        # train set and deploy (test) set 
        self.train_deploy_sets()
        # train decoupled classifiers
        self.opt_classifier()
        # train group-blind classifier
        self.opt_group_blind_classifier()

    
    def split_data(self):
        """
        split dataset based upon sensitive attribute
        """
        df_new = self.df.copy()
        
        # split dataset
        self.df_0 = df_new[df_new[self.S_name] == 0]
        self.df_1 = df_new[df_new[self.S_name] == 1]
        
        # get label
        self.Y_0 = self.df_0[self.Y_name].values
        self.Y_1 = self.df_1[self.Y_name].values
        
        # get features
        self.X_0 = self.df_0.drop([self.S_name, self.Y_name], axis=1).values
        self.X_1 = self.df_1.drop([self.S_name, self.Y_name], axis=1).values
        
        return 
    

    def split(self, X, Y, p = 0.7):
        """
        split (X, Y) into two sets based upon a percentage p
        """
        perm = np.random.RandomState(seed = self.rand_seed).permutation(len(Y))
        assert len(perm) == len(X)
        num_samp = int(p * len(perm))
        assert num_samp > 0
        
        X0 = X[perm[:num_samp]]
        X1 = X[perm[num_samp:]]
        
        Y0 = Y[perm[:num_samp]]
        Y1 = Y[perm[num_samp:]]
        
        return {"X0": X0, "X1": X1, "Y0": Y0, "Y1": Y1}
    
    
    def train_deploy_sets(self):
        """
        split into training set and deploying set
        """
        
        data_0 = self.split(self.X_0, self.Y_0)
        data_1 = self.split(self.X_1, self.Y_1)
        
        p_0 = np.random.RandomState(seed = self.rand_seed).permutation(len(self.Y_0))
        p_1 = np.random.RandomState(seed = self.rand_seed).permutation(len(self.Y_1))
        
        assert len(p_0) == len(self.X_0)
        assert len(p_1) == len(self.X_1)
        
        # number of training samples
        num_train_0 = int(0.7 * len(p_0))
        num_train_1 = int(0.7 * len(p_1))
        
        assert num_train_0 > 0
        assert num_train_1 > 0
        
        # split into train set and deploy set
        self.X_train_0 = data_0["X0"]
        self.Y_train_0 = data_0["Y0"]
        self.X_dep_0 = data_0["X1"]
        self.Y_dep_0 = data_0["Y1"]
        
        self.X_train_1 = data_1["X0"]
        self.Y_train_1 = data_1["Y0"]
        self.X_dep_1 = data_1["X1"]
        self.Y_dep_1 = data_1["Y1"]
    
        return 
    
    
    
    def opt_classifier(self):
        """
        train decoupled classifiers
        """
        # train a classifier on population 0
        X_train_0 = self.X_train_0
        Y_train_0 = self.Y_train_0
        assert len(X_train_0) > 0 and len(Y_train_0) > 0
        
        self.dec_0 = svm.SVC(gamma = 'scale', random_state = self.rand_seed)
        self.dec_0.fit(X_train_0, Y_train_0)
        
        # train a classifier on population 1
        X_train_1 = self.X_train_1
        Y_train_1 = self.Y_train_1
        assert len(X_train_1) > 0 and len(Y_train_1) > 0
        
        self.dec_1 = svm.SVC(gamma = 'scale', random_state = self.rand_seed)
        self.dec_1.fit(X_train_1, Y_train_1)
        
        return 
    

    
    def opt_group_blind_classifier(self):
        """
        train a group-blind classifier
        """
        # TODO: use 5-fold cross-validation to compute an optimal weight
        
        lamb = np.linspace(0, 1.0, 21)
        data_0 = self.split(self.X_train_0, self.Y_train_0)
        data_1 = self.split(self.X_train_1, self.Y_train_1)

        num_train_0 = len(data_0["X0"])
        num_train_1 = len(data_1["X0"])
        
        
        assert num_train_0 > 0 and num_train_1 > 0
        
        X_train = np.concatenate((data_1["X0"], data_0["X0"]), axis=0)
        Y_train = np.concatenate((data_1["Y0"], data_0["Y0"]), axis=0)
        
        X_val_0 = data_0["X1"]
        Y_val_0 = data_0["Y1"]
        X_val_1 = data_1["X1"]
        Y_val_1 = data_1["Y1"]
        
        group_blind = svm.SVC(gamma = 'scale', random_state = self.rand_seed)
        weight = np.zeros(len(Y_train))
        acc = []
        
        for l in lamb:
            for i in range(len(Y_train)): 
                if i < num_train_1:
                    weight[i] = l
                else:
                    weight[i] = 1.0 - l
            group_blind.fit(X_train, Y_train, sample_weight = weight)
            temp0 = np.abs(group_blind.predict(X_val_0).astype(float) - Y_val_0)
            val_acc_0 = sum(temp0) / len(temp0)
            temp1 = np.abs(group_blind.predict(X_val_1).astype(float) - Y_val_1)
            val_acc_1 = sum(temp1) / len(temp1)
            
            acc.append(max(val_acc_0, val_acc_1))
        
        assert len(acc) == len(lamb)
        l = lamb[np.argmin(acc)]
        for i in range(len(Y_train)): 
            if i < num_train_1:
                weight[i] = l
            else:
                weight[i] = 1.0 - l
        
        group_blind.fit(X_train, Y_train, sample_weight = weight)
        self.group_blind = group_blind
        
        return 
            
    

    def comp_approx_error(self):
        """
        compute accuracy of decoupled classifiers on each population
        """
        assert callable(self.dec_0.predict)
        assert callable(self.dec_1.predict)
        temp0 = np.abs(self.dec_0.predict(self.X_dep_0).astype(float) - self.Y_dep_0)
        self.acc_0 = sum(temp0)/len(temp0)
        
        temp1 = np.abs(self.dec_1.predict(self.X_dep_1).astype(float) - self.Y_dep_1)
        self.acc_1 = sum(temp1)/len(temp1)
        
        return
    
    def comp_group_blind_loss(self):
        """
        compute accuracy of group-blind classifier on each population
        """
        assert callable(self.dec_0.predict)
        assert callable(self.dec_1.predict)
        
        temp0 = np.abs(self.group_blind.predict(self.X_dep_0).astype(float) - self.Y_dep_0)
        self.acc_group_blind_0 = sum(temp0)/len(temp0)
        
        temp1 = np.abs(self.group_blind.predict(self.X_dep_1).astype(float) - self.Y_dep_1)
        self.acc_group_blind_1 = sum(temp1)/len(temp1)
        
        return 
    
    
    def comp_eps_split(self):
        """
        compute the benefit of splitting
        """
        self.comp_approx_error()
        self.comp_group_blind_loss()
        # compute the benefit of splitting
        self.eps_split = max(self.acc_group_blind_0, self.acc_group_blind_1) - max(self.acc_0, self.acc_1)
        self.eps_split = max(self.eps_split, 0)
        
        return 


    def comp_disc(self):
        """
        compute the disagreement between two optimal classifiers
        """
        
        # compute discrepancy using minority data
        assert callable(self.dec_0.predict)
        assert callable(self.dec_1.predict)
        
        temp0 = np.abs(self.dec_0.predict(self.X_dep_0).astype(float) 
                       - self.dec_1.predict(self.X_dep_0).astype(float))
        self.disc0 = sum(temp0)/len(temp0)
        
        # compute discrepancy using majority data
        temp1 = np.abs(self.dec_0.predict(self.X_dep_1).astype(float) 
                       - self.dec_1.predict(self.X_dep_1).astype(float))
        self.disc1 = sum(temp1)/len(temp1)
        
        return 
    
    # TODO: improve this function
    def comp_f_div(self):
        """
        compute chi-square divergence and total variation distance between unlabeled distributions
        """
        X_0 = self.X_0
        X_1 = self.X_1
        X = np.concatenate((X_0, X_1), axis=0)
        alphabet = np.unique(X, axis=0)
        
        P_0 = []
        P_1 = []
        for mass_point in alphabet:
            prob_0 = np.sum((X_0 == mass_point).all(axis = 1)) / float(len(X_0))
            prob_1 = np.sum((X_1 == mass_point).all(axis = 1)) / float(len(X_1))
            # if probability mass is zero, 
            # we add a small constant to aviod chi-square-divergence = infinity
            if prob_0 == 0:
                prob_0 = 1e-4
            if prob_1 == 0:
                prob_1 = 1e-4
            P_0.append(prob_0)
            P_1.append(prob_1)
        
        # normalize the probability vectors
        P_0 = P_0 / sum(P_0)
        P_1 = P_1 / sum(P_1)
        
        assert np.isclose(1.0, sum(P_0))
        assert np.isclose(1.0, sum(P_1))
        
        chi_sq_0 = 0
        chi_sq_1 = 0
        tv = 0
        for i in range(len(alphabet)):
            # compute chi-square-divergence(P_{X|S=0} || P_{X|S=1})
            if P_1[i] == 0:
                chi_sq_0 = np.inf
            else:
                chi_sq_0 += P_1[i] * ((P_0[i] / P_1[i])**2 - 1.0)
            
            # compute chi-square-divergence(P_{X|S=1}|| P_{X|S=0})
            if P_0[i] == 0:
                chi_sq_1 = np.inf
            else:
                chi_sq_1 += P_0[i] * ((P_1[i] / P_0[i])**2 - 1.0)
            
            # compute total-variation-distance(P_{X|S=1} || P_{X|S=0}) 
            # Note that the total-variation-distance is symmetric
            tv += np.abs(P_1[i] - P_0[i])
        tv = tv / 2.0
        assert tv >= 0 and chi_sq_0 >= 0 and chi_sq_1 >= 0
        
        self.chi_sq_0 = chi_sq_0
        self.chi_sq_1 = chi_sq_1
        self.tv = tv        
        
        return 
    
    def bounds(self):
        """
        compute upper and lower bounds for the benefit of splitting
        """
        self.comp_disc()
        self.comp_f_div()
        self.ub = min(self.disc0, self.disc1)
        self.lb = max((self.disc0 / (np.sqrt(self.chi_sq_0 + 1.0) + 1.0))**2, 
                      (self.disc1 / (np.sqrt(self.chi_sq_1 + 1.0) + 1.0))**2)
        
        return 
    
    def get_bounds(self):
        """
        return upper and lower bounds
        """
        self.bounds()
        ub = self.ub
        lb = self.lb
        
        return {"ub": ub, "lb": lb}
    
    def get_ben_split(self):
        """
        return
        :eps_split: the benefit of splitting
        :dec_0: decoupled classifiers' accuracy on population 0
        :dec_1: decoupled classifiers' accuracy on population 1
        :group_blind_0: group-blind classifier accuracy on population 0
        :group_blind_1: group-blind classifier accuracy on population 1
        """
        self.comp_eps_split()
        eps_split = self.eps_split
        acc_0 = self.acc_0
        acc_1 = self.acc_1
        acc_group_blind_0 = self.acc_group_blind_0
        acc_group_blind_1 = self.acc_group_blind_1
        
        return {"eps_split": eps_split, "dec_0": acc_0, "dec_1": acc_1, 
                "group_blind_0": acc_group_blind_0, "group_blind_1": acc_group_blind_1}
    
    def get_div(self):
        """
        return chi-square divergence and total variation distance
        :chi_sq_0: D_{\chi^2}(P_{X|S=0} || P_{X|S=1})
        :chi_sq_1: D_{\chi^2}(P_{X|S=1} || P_{X|S=0})
        :tv: D_{tv}(P_{X|S=0}, P_{X|S=1})
        """
        chi_sq_0 = self.chi_sq_0
        chi_sq_1 = self.chi_sq_1
        tv = self.tv
        
        return {"chi_sq_0": chi_sq_0, "chi_sq_1": chi_sq_1, "tv": tv}

    def get_unlabel_dataset(self):
        """
        return two unlabeled datasets
        """
        X_0 = self.X_0
        X_1 = self.X_1
        
        return {"X_0": X_0, "X_1": X_1}



def preprocess_binary(df, S_name, Y_name):
    """
    convert categorical features/sentive attribute/label to binary
    """
    df_new = df.copy()
    
    # feature names
    feat_name = df_new.columns.values.tolist()
    
    for feat in feat_name:
        feat_type = df_new[feat].dtype
        # if categorical features or sensitive attribute or label,
        # convert to binary feature.
        if feat_type == "object" or feat == S_name or feat == Y_name:
            feat_count = df_new[feat].value_counts()
            df_new[feat] = df_new[feat].replace(feat_count.index[1:], np.nan)
            df_new[feat] = df_new[feat].notnull() * 1
    
    return df_new




data_info = {'ID': [21, 23, 26, 31, 50, 151, 155, 183, 184, 292, 333, 334, 335, 351, 354, 375,
            469, 475, 679, 720, 741, 825, 826, 872, 881, 915, 923, 934, 959, 983, 991, 
            1014, 1169, 1216, 1217, 1218, 1235, 1236, 1237, 1470, 1481, 1483, 1498, 
            1557, 1568, 4135, 4552], 
            "sensitive attribute": ['buying', 'Wifes_education', 'parents', 'checking_status', 'top-left-square', 'day', 's1', 'Sex', 'white_king_row', 'Y',
                                  'class', 'class', 'class', 'Y', 'Y', 'speaker', 'DMFT.Begin', 'Time_of_survey', 'sleep_state', 'Sex',
                                  'sleep_state', 'RAD', 'Occasion', 'RAD', 'x3', 'SMOKSTAT', 'isns', 'family_structure', 'parents', 'Wifes_education',
                                  'buying', 'DMFT.Begin', 'Airline', 'click', 'click', 'click', 'elevel', 'size', 'size', 'V2',
                                  'V3', 'V1', 'V5', 'V1', 'V1', 'RESOURCE', 'V1'], 
            "label": ['class', 'Contraceptive_method_used', 'class', 'class', 'Class', 'class', 'class', 'Class_number_of_rings', 'game', 'X1',
                                  'attr1', 'attr1', 'attr1', 'X1', 'X1', 'utterance', 'Prevention', 'Political_system', 'temperature', 'binaryClass',
                                  'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass', 'binaryClass',
                                  'binaryClass', 'binaryClass', 'Delay', 'impression', 'impression', 'impression', 'class', 'class', 'class', 'Class',
                                  'Class', 'Class', 'Class', 'Class', 'Class', 'target', 'V17'],
            "link": ["https://www.openml.org/data/get_csv/" + string for string in ["21/dataset_21_car.arff", "23/dataset_23_cmc.arff", "26/dataset_26_nursery.arff", "31/dataset_31_credit-g.arff", "50/dataset_50_tic-tac-toe.arff", "2419/electricity-normalized.arff", "2423/pokerhand-normalized.arff", "3620/dataset_187_abalone.arff", "3621/dataset_188_kropt.arff", "49822/australian.arff", 
                                                                             "52236/phpAyyBys", "52237/php4fATLZ", "52238/phphZierv", "52254/php89ntbG", "52257/phpQfR7GF", "52415/JapaneseVowels.arff", "52581/analcatdata_dmft.arff", "52587/analcatdata_germangss.arff", "52979/rmftsa_sleepdata.arff", "53254/abalone.arff",
                                                                             "53275/rmftsa_sleepdata.arff", "53359/boston_corrected.arff", "53360/sensory.arff", "53406/boston.arff", "53415/mv.arff", "53449/plasma_retinol.arff", "53457/visualizing_soil.arff", "53468/socmob.arff", "53493/nursery.arff", "53517/cmc.arff",
                                                                             "53525/car.arff", "53548/analcatdata_dmft.arff", "66526/phpvcoG8S", "183030/phppCF8Zy", "183039/phpLV1N3m", "183150/phpqZOQcc", "520800/Agrawal1.arff", "520801/Stagger1.arff", "520802/Stagger2.arff", "1586239/phpce61nO",
                                                                             "1590570/php7zhUPY", "1590940/phpH4DHsK", "1592290/phpgNaXZe", "1593753/phpfUae7X", "1675984/phpfrJpBS", "1681098/phpmPOD5A", "1798821/php0mZlkF"]]}
datasets = pd.DataFrame(data = data_info)


def preprocess_openml(df, ID, S_name, Y_name):
    df_new = df.copy()    
    df_new = preprocess_binary(df_new, S_name, Y_name)
    
    return df_new


# dataset index
num_dataset = 0

# read csv file
df = pd.io.parsers.read_csv(datasets["link"][num_dataset])
# dataset ID
ID = datasets["ID"][num_dataset]
# sensitive attribute
S_name = datasets["sensitive attribute"][num_dataset]
# label
Y_name = datasets["label"][num_dataset]


df_new = preprocess_openml(df, ID, S_name, Y_name)

# call class
obj_BenefitSplit = BenefitSplit(df_new, S_name, Y_name)

# compute bounds
val_bound = obj_BenefitSplit.get_bounds()

# compute the benefit of splitting
val_ben_split = obj_BenefitSplit.get_ben_split()

print(num_dataset)
print(val_ben_split)
print(val_bound)
print(obj_BenefitSplit.get_div())











'''


# compute chi-square divergence


data = obj_BenefitSplit.get_unlabel_dataset()
X_1 = data["X_1"]
X_0 = data["X_0"]



def cs_divergence(x1, x0, n_input, n_neuron, n_output):
    h1 = H_Net(x1, n_input, name="HNet", structure = [n_neuron, n_neuron, n_output])
    h0 = H_Net(x0, n_input, name="HNet", structure = [n_neuron, n_neuron, n_output])
    
#     h1 = H_Net(x1, n_input, name="HNet")
#     h0 = H_Net(x0, n_input, name="HNet")
    
    return 2*tf.reduce_mean(h1) - tf.reduce_mean(tf.math.square(h0)) - 1, h1, h0
#     return tf.reduce_mean(h1) - tf.math.log(tf.reduce_mean(tf.math.exp(h0))), h1, h0
    

def CSD_estimate(X0, X1, n_epoch, n_neuron, lr):
    assert X0.shape[1] == X1.shape[1]
    n_input = X0.shape[1]
    n_output = 1
    n_train = 800
    
    sess = tf.InteractiveSession()
    x1 = tf.placeholder(tf.float32, [None, n_input], name='X1')
    x0 = tf.placeholder(tf.float32, [None, n_input], name='X0')
    
    l, h1, h0 = cs_divergence(x1, x0, n_input, n_neuron, n_output)
    loss = -l
    solver = tf.train.AdagradOptimizer(lr).minimize(loss)

    tf.global_variables_initializer().run()
    
    print('epoch\t loss')
    for epoch in range(n_epoch):
        _, current_loss = sess.run([solver, loss], feed_dict={x1: X1[:n_train], x0: X0[:n_train]})
        if epoch % 100 == 0:
            print('{}\t {:.4f}'.format(epoch, current_loss))
        
    CSD, H1, H0 = sess.run([loss, h1, h0], feed_dict={x1: X1[:n_train], x0: X0[:n_train]})
    CSD_test, _, _ = sess.run([loss, h1, h0], feed_dict={x1: X1[n_train:], x0: X0[n_train:]})
    print('Chi-Squared Divergence: {:.4f}'.format(CSD))
    print('Chi-Squared Divergence (Test): {:.4f}'.format(CSD_test))
    sess.close()
    return H1, H0

n_epoch = 1000
n_neuron = 50
lr = 1e-1
h1, h0 = CSD_estimate(X_1, X_0, n_epoch, n_neuron, lr)

'''








