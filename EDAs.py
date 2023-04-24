from deap import tools
import foolbox as fb
import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical
from netTraining import train, initialize_variables, back_drive_net
from structure import initialize_back_variables, close
import warnings
import copy
import tensorflow as tf
warnings.filterwarnings("ignore")


class EDA(object):
    def __init__(self, pop_sz, ind_size, evl, method, offspring, **kwargs):
        self.pop_size = pop_sz
        self.ind_size = ind_size
        self.eval = evl
        self.high_scale = None
        self.low_scale = None
        self.best = None
        self.low_scale = None
        self.high_scale = None
        self.sel_method = method
        self.offspring = offspring
        self.args = kwargs

    def initial_population(self, ind_init):
        pop = create_population(self.ind_size, self.pop_size)

        ind_list = list(map(ind_init, pop))
        fitnesses = list(map(self.eval, pop))

        for ind, fit in zip(ind_list, fitnesses):
            ind.fitness.values = (fit,)

        return ind_list

    def best_inds(self, population, method):
        if method == 0:
            best = tools.selBest(population, k=int(self.pop_size*(1-self.offspring)))  # Get the best half of individuals
        else:
            best = tools.selTournament(population, k=int(self.pop_size*(1-self.offspring)), tournsize=self.args["tour_size"])

        return best

    def new_population(self, new_pop, ind_init):

        ind_list = list(map(ind_init, new_pop))  # Transform the population to DEAP format (with fitness, ...)

        # Compute and add the fitness values to the individuals

        fitnesses = list(map(self.eval, new_pop))
        for ind, fit in zip(ind_list, fitnesses):
            ind.fitness.values = (fit,)
        return ind_list


def create_individual(ind_size):
    """
    :param ind_size: Size of the solution
    :return:
    """

    ind = []
    for i in range(ind_size):
        ind.append(np.random.rand())

    return ind


def create_population(ind_size, pop_sz, index=False):
    """
    :param ind_size: Amount of variables in the function
    :param pop_sz: Amount of solutions in the population
    :param index: Booleans, whether the solutions will correspond to the first integers (True) or not
    :return: the population according to the mentioned criteria
    """
    population = []
    for fixed_ind in range(pop_sz):
        if index:
            ind = create_individual(ind_size)
        else:
            ind = create_individual(ind_size)
        population.append(ind)
    population = np.array(population)

    return population


class UMDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)
        self.mu = None
        self.sigma = None
        self.best = None

    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            pop = np.random.normal(self.mu, self.sigma, (int(self.pop_size*self.offspring), self.ind_size))

            pop = np.clip(pop, 0, 1)
            ind_list = self.new_population(pop, ind_init) + self.best  # The best individuals in the previous generation and the new ones

        return ind_list

    def update(self, population):

        self.best = self.best_inds(population, self.sel_method, )
        self.mu = np.mean(self.best, axis=0)
        self.sigma = np.sqrt(np.var(self.best, axis=0))


class Random(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)
        self.best = None

    # This function generates individuals
    def generate(self, ind_init, gen):

        if self.best is None:
            ind_list, _ = self.initial_population(ind_init)
        else:
            pop = create_population(self.ind_size, self.pop_size-1)

            ind_list, _ = self.new_population(pop, ind_init) + [self.best]  # The best individuals in the previous generation and the new ones

        return ind_list

    def update(self, population):
        self.best = self.best_inds(population, self.sel_method)[0]


class AdvEDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring,
                 adv_method_type, treatment_bad, treatment_ret_pred1, treatment_ret_pred0,
                 trick_network, extra_layers, early_stopping, retain_w, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)

        self.trick_network = trick_network
        self.extra_layers = extra_layers
        self.early_stopping = early_stopping
        print(trick_network, extra_layers, early_stopping)

        self.best = None
        self.worst = None
        self.foolbox_model = None
        self.retain = []
        self.retain_f = []
        self.eval_function = eval_function
        self.n_implemented_attacks = 16
        self.retain_w = retain_w

        self.adv_method_type = adv_method_type
        self.treatment_bad = treatment_bad
        self.treatment_ret_pred1 = treatment_ret_pred1
        self.treatment_ret_pred0 = treatment_ret_pred0
        self.patience = 3
        self.model = self.initialize_nn(num_variables, num_variables*3, num_variables*2)
        self.w_save = self.model.get_weights()

    # This function generates individuals
    def old_generate(self, ind_init):
        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []
            for bad in self.worst:
                pred = np.argmax(self.model.predict(np.array([bad])), axis=1)[0]
                new_pop += [self.generate_adv_example(self.foolbox_model, bad, pred)]

            # print("Length is ", len(new_pop), " and total ", len(ind_init))
            ind_list = self.new_population(np.array(new_pop), ind_init) + self.new_population(np.array(self.retain), ind_init)

        return ind_list

        # This function generates individuals
    def generate(self, ind_init):
        eta = 0.1     # strength of polynomial mutation
        low = 0       # lower bound for mutation
        up = 1        # upper bound for mutation
        indpb = 0.5   # mutation probability
        
        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []

            for bad in self.worst:
                # pred = np.argmax(self.model.predict(np.array([bad])), axis=1)[0]
                # print(bad,"pred ",pred)
                if self.treatment_bad == 0:
                    bad = np.array(create_individual(len(bad)))
                elif self.treatment_bad == 1:
                    pass      # We do nothing, the solution is perturbed by AdvModel                
                new_pop += [self.generate_adv_example(self.foolbox_model, bad, 0)]

            for ret in self.retain:
                pred = np.argmax(self.model.predict(np.array([ret])), axis=1)[0]
                # print(ret,"pred-ret ",pred,len(ret))
                if pred == 1:
                    if self.treatment_ret_pred1 == 0:
                        ret = tools.mutPolynomialBounded(ret, eta, low, up, indpb)[0]
                    elif self.treatment_ret_pred1 == 1:
                        pos = np.random.randint(len(self.best))
                        ret = copy.copy(self.best[pos])
                        ret = tools.mutPolynomialBounded(ret, eta, low, up, indpb)[0]
                    elif self.treatment_ret_pred1 == 2:
                        ret = np.array(create_individual(len(ret)))
                elif pred == 0:
                    if self.treatment_ret_pred0 == 0:
                        ret = tools.mutPolynomialBounded(ret, eta, low, up, indpb)[0]
                    elif self.treatment_ret_pred0 == 1:
                        pos = np.random.randint(len(self.best))
                        ret = copy.copy(self.best[pos])
                        ret = tools.mutPolynomialBounded(ret, eta, low, up, indpb)[0]
                    elif self.treatment_ret_pred0 == 2:
                        ret = np.array(create_individual(len(ret)))
                    elif self.treatment_ret_pred0 == 3:
                        pass
                new_pop += [self.generate_adv_example(self.foolbox_model, ret, 0)]

            # print("Length is ", len(new_pop), " and total ", len(ind_init))
            # ind_list = self.new_population(np.array(new_pop), ind_init)
            ind_list = self.new_population(np.array(new_pop), ind_init) + self.new_population(np.array(self.best), ind_init)

        return ind_list

    @staticmethod
    def define_attack(model, sel):
        # start_time = time.time()
        attack = None
        if sel == 0:
            attack = fb.v1.attacks.GradientAttack(model)
        elif sel == 1:
            attack = fb.v1.attacks.DecoupledDirectionNormL2Attack(model)  # Also good
        elif sel == 2:
            attack = fb.v1.attacks.EADAttack(model)
        elif sel == 3:
            attack = fb.v1.attacks.GradientSignAttack(model)  # Fast
        elif sel == 4:
            attack = fb.v1.attacks.AdamL1BasicIterativeAttack(model, distance=fb.distances.MAE)
        elif sel == 5:
            attack = fb.v1.attacks.RandomPGD(model, distance=fb.distances.Linfinity)  # Slow
        elif sel == 6:
            attack = fb.v1.attacks.RandomStartProjectedGradientDescentAttack(model, distance=fb.distances.Linfinity)
        elif sel == 7:
            attack = fb.v1.attacks.AdamL2BasicIterativeAttack(model)
        elif sel == 8:
            attack = fb.v1.attacks.AdditiveUniformNoiseAttack(model)
        elif sel == 9:
            attack = fb.v1.attacks.MomentumIterativeAttack(model, distance=fb.distances.Linfinity)
        elif sel == 10:
            attack = fb.v1.attacks.DeepFoolAttack(model, criterion=fb.criteria.Misclassification())  # Fast
        elif sel == 11:
            attack = fb.v1.attacks.LBFGSAttack(model)  # Slow
        elif sel == 12:
            attack = fb.v1.attacks.L2BasicIterativeAttack(model)
        elif sel == 13:
            attack = fb.v1.attacks.SparseFoolAttack(model)
        elif sel == 14:
            attack = fb.v1.attacks.AdditiveGaussianNoiseAttack(model)
        elif sel == 15:
            attack = fb.v1.attacks.BlendedUniformNoiseAttack(model)
        # elapsed_time = time.time() - start_time
        return attack

    @staticmethod
    def apply_attack(attack, sample, sample_label):
        warnings.filterwarnings("error")
        while True:
            try:
                result = attack(sample, sample_label)
                break
            except ValueError:
                # print("Oops!  That was no valid number.  Try again...")
                result = None
                break
            except AssertionError:
                result = None
                break
            except IndexError:
                result = None
                break
            except UserWarning:
                result = None
                break
            except RuntimeWarning:
                result = None
                break
            except 'WARNING:root:exponential search failed':
                result = None
                break
        return result

    def generate_adv_example(self, model, sample, sample_label):  # , target_label=None):
        if self.adv_method_type >= self.n_implemented_attacks:
            sel = np.random.randint(self.n_implemented_attacks)
        else:
            sel = self.adv_method_type
        attack = self.define_attack(model, sel)
        
        # print("Samp before attack",sample)
        # sample_eval = self.eval_function(sample)

        result = self.apply_attack(attack, sample, sample_label)
        # print("Samp after attack",result)
        
        if result is None:
            # print(result)
            result = copy.copy(sample)
            # print("sel ",sel, sample_eval,-1,-1,elapsed_time)
        else:
            pass  # result_eval = self.eval_function(result)
            # print("sel ",sel, sample_eval, result_eval, sample_eval-result_eval,elapsed_time)
          
        return result

    def update(self, population):
        X_train, y_train = self.select_best_worst(population, self.sel_method)

        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs.

        if self.early_stopping == 1:
            prop_validation = 0.2
            n_train = len(X_train)
            n_val = int(prop_validation * n_train)
            rnd_perm = np.random.permutation(n_train)
            val_data = X_train[rnd_perm[:n_val], :]
            val_labels = y_train[rnd_perm[:n_val], :]
            train_data = X_train[rnd_perm[n_val:], :]
            train_labels = y_train[rnd_perm[n_val:], :]

            validation_data = (val_data, val_labels)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
            self.model.fit(train_data, train_labels, batch_size=10, epochs=250, verbose=0, callbacks=[callback],
                           validation_data=validation_data, shuffle=True, class_weight=None, sample_weight=None)
        elif self.early_stopping == 0:
            self.model.fit(X_train, y_train, batch_size=10, epochs=250, verbose=0, callbacks=[],
                           validation_data=[], shuffle=True, class_weight=None, sample_weight=None)

        # print("Whether network is tricked or not", self.trick_network)
        if self.trick_network == 1:
            n_layers = len(self.model.layers)
            last_layer_weights = self.model.layers[n_layers - 1].get_weights()[0]
            last_layer_biases = self.model.layers[n_layers - 1].get_weights()[1]
            new_weights = np.zeros((last_layer_weights.shape))
            new_biases = last_layer_biases  # np.zeros((last_layer_biases.shape))
            trials = 10
            npred = len(self.best)
            # print("Weights n-2",self.model.layers[n_layers-2].get_weights())
            # print("Weights n-1",self.model.layers[n_layers-1].get_weights())
            i = 0

            while (i < trials and npred > len(self.best) - 1):
                alpha = 0.01 * i
                new_weights[:, 0] = last_layer_weights[:, 0] * (1 + alpha)
                new_weights[:, 1] = last_layer_weights[:, 1] * (1 - alpha)
                # print(new_weights)
                self.model.layers[n_layers - 1].set_weights([new_weights, new_biases])
                npred = 0
                for b in self.best:
                    pred = np.argmax(self.model.predict(np.array([b])), axis=1)[0]
                    npred += pred
                # nretain = 0
                # for b in self.retain:
                #    pred = np.argmax(self.model.predict(np.array([b])), axis=1)[0]
                #    nretain += pred
                # print('pred ',i,npred,nretain)
                i = i + 1
        self.foolbox_model = fb.models.TensorFlowModel.from_keras(self.model, bounds=(0, 1))
        
    def select_best_worst(self, population, method):

        fitness = np.array([-x.fitness.wvalues[0] for x in population])       
        individuals = np.array([np.array(population[i]) for i in range(len(population))])
        best_sols = np.argsort(fitness)
           
        # We select the Trunc percentage of best X
        self.best = individuals[best_sols[:self.args["sel_pop_size"]], :]
        # self.retain = individuals[best_sols[:(len(population)-self.args["sel_pop_size"])], :]
        self.retain = individuals[best_sols[self.args["sel_pop_size"]:(len(population)-self.args["sel_pop_size"])], :]
        # self.retain_f = fitness[:best_sols[-self.args["sel_pop_size"]]]
        # We select the Trunc percentage of worst X
        self.worst = individuals[best_sols[-self.args["sel_pop_size"]:], :]       

        X_train = np.vstack(([self.best, self.worst]))
        y_train = np.hstack((np.ones(self.args["sel_pop_size"]), np.zeros(len(self.worst))))
        y_train = to_categorical(y_train)

        # X_train = np.vstack(([self.best, self.worst]))
        # y_train = np.hstack((np.ones(self.args["sel_pop_size"]), np.zeros(self.args["sel_pop_size"])))
        # y_train = to_categorical(y_train)

        return X_train, y_train

    def initialize_nn(self, x_dim, z_dim_1, z_dim_2):
        # initialize model
        model = keras.models.Sequential()
        # add input layer
        model.add(keras.layers.Dense(units=z_dim_1, input_shape=(x_dim,), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid', name='First'))
        # add hidden layer
        model.add(keras.layers.Dense(units=z_dim_2, input_dim=z_dim_1, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid',name='Second'))
        for l in range(self.extra_layers):
            # add hidden layer
            model.add(keras.layers.Dense(units=z_dim_1, input_dim=z_dim_2, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))
            # add hidden layer
            model.add(keras.layers.Dense(units=z_dim_2, input_dim=z_dim_1, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))


        #model.add(keras.layers.Dropout(0.2, input_shape=(z_dim_2,)))        
        model.add(keras.layers.Dense(units=2, input_dim=z_dim_2, kernel_initializer='zeros', bias_initializer='zeros', activation='softmax',name='Final'))
        # define SGD optimizer
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        #sgd_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        #tf.train.FtrlOptimizer(0.01) #.minimize(loss)
        # compile model
        model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

class BackDriveEDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, outputs, method, offspring, retain_ws, retain_inds, sol_disc, train_m, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)
        self.sess, self.x, self.y, self.weights, self.hidden_weights, self.biases, self.hidden_biases, self.net, self.loss, self.optimizer, _, _, _, _ = initialize_variables(input_size=num_variables, output_size=outputs, h_neurons=self.args["h_neurons"] if "h_neurons" in self.args else 50, batch_size=self.args["batch_size"] if "batch_size" in self.args else 50, learning_rate=self.args["lr"] if "lr" in self.args else 0.001, sigma=self.args["sigma"] if "sigma" in self.args else 15, variable_groups=None, structure=0, delete=True)
        self.individual, self.ind_initializer, self.clipped_ind, self.target, self.back_weights, self.back_hidden_weights, self.back_biases, self.back_hidden_biases, self.back_net, self.perf_loss, self.apply_grads, self.grads, _, self.sum_back, _, _, _ = initialize_back_variables(sess=self.sess, input_size=num_variables, output_size=outputs, h_neurons=self.args["h_neurons"] if "h_neurons" in self.args else 50, sigma=self.args["sigma"] if "sigma" in self.args else 15, variable_groups=None, structure=0, number=pop_sz*(1 if sol_disc == 0 else 20))
        self.best = None
        self.retain_ws = retain_ws
        self.retain_inds = retain_inds
        self.sol_disc = sol_disc
        self.rand_pop = None
        self.rand_fit = None
        self.elite_pop = None
        self.elite_fit = None
        self.historic_pop = None
        self.historic_fit = None
        self.train_m = train_m

    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
            self.rand_fit = np.array([-x.fitness.wvalues[0] for x in ind_list])
            self.rand_pop = np.array(ind_list)
            self.elite_pop = self.rand_pop
            self.elite_fit = self.rand_fit
            self.historic_pop = self.rand_pop
            self.historic_fit = self.rand_fit
        else:
            w, h_w, b, h_b = self.sess.run([self.weights, self.hidden_weights, self.biases, self.hidden_biases])
            if self.retain_inds == 0:
                self.sess.run(self.ind_initializer)
            back_drive_net(self.sess, self.args["convergence"] if "convergence" in self.args else 0.01, self.best, self.target, self.apply_grads, self.perf_loss, self.back_weights, self.back_hidden_weights, self.back_biases, self.back_hidden_biases, w, h_w, b, h_b, 500, self.clipped_ind, self.back_net)
            ind = np.clip(self.sess.run(self.individual), 0, 1)
            if ind.shape[0] > self.pop_size:
                preds = self.sess.run(self.net, feed_dict={self.x: ind})[:, 0]
                print(preds, self.pop_size/ind.shape[0], np.nanquantile(preds, self.pop_size/ind.shape[0]))
                print(np.sum(preds <= np.nanquantile(preds, self.pop_size/ind.shape[0])))
                ind = ind[preds <= np.nanquantile(preds, self.pop_size/ind.shape[0])]

            ind_list = self.new_population(ind, ind_init)
        return ind_list

    def update(self, population, percentile, noise):

        fitness = np.array([-x.fitness.wvalues[0] for x in population])
        individuals = np.array([np.array(population[i]) for i in range(len(population))])

        self.historic_pop = np.concatenate((individuals, self.historic_pop[:(10000-individuals.shape[0])]))
        self.historic_fit = np.concatenate((fitness, self.historic_fit[:(10000 - fitness.shape[0])]))

        self.elite_fit = np.concatenate((self.elite_fit, fitness))
        self.elite_pop = np.concatenate((self.elite_pop, individuals))

        indices = self.elite_fit <= np.nanquantile(self.elite_fit, self.pop_size/self.elite_pop.shape[0])

        self.elite_pop = self.elite_pop[indices]
        self.elite_fit = self.elite_fit[indices]
        if self.train_m == 1:
            individuals = np.concatenate((individuals, self.rand_pop, self.elite_pop, self.historic_pop))
            fitness = np.concatenate((fitness, self.rand_fit, self.elite_fit, self.historic_fit))

        fitness = fitness - np.min(fitness)
        fitness = fitness + 0.01*np.max(fitness)
        fitness = fitness / np.max(fitness)

        outs = np.array([np.log(fitness), fitness, np.sqrt(fitness), np.power(fitness, 2), np.sin(fitness)])[:self.net.shape[1]].T

        outs = outs - np.min(outs, axis=0)
        outs = outs + 0.01 * np.max(outs, axis=0)
        outs = outs / np.max(outs, axis=0)
        self.best = np.percentile(outs, percentile, axis=0).reshape(-1, self.net.shape[1])

        if noise == 1:
            self.best = np.array([np.random.normal(loc=var, scale=0.01, size=self.pop_size) for var in self.best[0]]).T

        train(net_input=individuals, net_output=outs, start=False, sess=self.sess, x=self.x, y=self.y, net=self.net, loss=self.loss, optimizer=self.optimizer, sum_ford=None, saver=None, writer=None, log=None, epoch=self.args["epochs"] if "epochs" in self.args else 1, batch_size=self.args["batch_size"] if "batch_size" in self.args else 50, display_step=100000, save_step=10000, groups=None, structure=0, init=self.retain_ws == 0)

    def close(self):
        close(self.sess)


class Old_AdvEDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)

        self.model = self.initialize_nn(num_variables, num_variables*3, num_variables*2)
        self.best = None
        self.worst = None
        self.foolbox_model = None
        self.retain = []
        self.retain_f = []
    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []
            for bad in self.worst:
                pred = np.argmax(self.model.predict(np.array([bad])), axis=1)[0]
                new_pop += [self.generate_adv_example(self.foolbox_model, bad, pred)]

            ind_list = self.new_population(np.array(new_pop), ind_init)

        return ind_list

    @staticmethod
    def generate_adv_example(model, sample, sample_label, target_label=None):

        if target_label is None:
            attack = fb.v1.attacks.FGSM(model)
        else:
            criterion = fb.criteria.TargetClass(target_label)
            attack = fb.v1.attacks.FGSM(model, criterion)

        return attack(sample, sample_label)

    def update(self, population):
        X_train, y_train = self.select_best_worst(population, self.sel_method)
        self.model.fit(X_train, y_train, batch_size=100, epochs=250, verbose=0, callbacks=[], validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
        self.foolbox_model = fb.models.TensorFlowModel.from_keras(self.model, bounds=(0, 1))

    def select_best_worst(self, population, method):

        fitness = np.array([-x.fitness.wvalues[0] for x in population])
        individuals = np.array([np.array(population[i]) for i in range(len(population))])

        best_sols = np.argsort(fitness)

        # We select the Trunc percentage of best X
        self.best = individuals[best_sols[:self.args["sel_pop_size"]], :]
        self.retain = individuals[:best_sols[-self.args["sel_pop_size"]], :]
        self.retain_f = fitness[:best_sols[-self.args["sel_pop_size"]]]
        # We select the Trunc percentage of worst X
        self.worst = individuals[best_sols[-self.args["sel_pop_size"]:], :]

        X_train = np.vstack(([self.best, self.worst]))
        y_train = np.hstack((np.ones(self.args["sel_pop_size"]), np.zeros(self.args["sel_pop_size"])))
        y_train = to_categorical(y_train)

        return X_train, y_train

    @staticmethod
    def initialize_nn(x_dim, z_dim_1, z_dim_2):
        # initialize model
        model = keras.models.Sequential()
        # add input layer
        model.add(keras.layers.Dense(units=z_dim_1, input_shape=(x_dim,), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))
        # add hidden layer
        model.add(keras.layers.Dense(units=z_dim_2, input_dim=z_dim_1, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))
        # add output layer
        model.add(keras.layers.Dense(units=2, input_dim=z_dim_2, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))
        # define SGD optimizer
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        # compile model
        model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


class UMDA_UMDA(AdvEDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring,
                 adv_method_type, treatment_bad, treatment_ret_pred1, treatment_ret_pred0,
                 trick_network, extra_layers, early_stopping, **kwargs):

        super().__init__(eval_function, pop_sz, num_variables, method, offspring,
                         adv_method_type, treatment_bad, treatment_ret_pred1, treatment_ret_pred0,
                         trick_network, extra_layers, early_stopping, **kwargs)
        self.mu = None
        self.sigma = None
        self.best = None

        # print("It arrived here")
        # pass

    def update(self, population):
        X_train, y_train = self.select_best_worst(population, self.sel_method)
        self.mu = np.mean(self.best, axis=0)
        self.sigma = np.sqrt(np.var(self.best, axis=0))

    def generate(self, ind_init):
        eta = 0.1  # strength of polynomial mutation
        low = 0  # lower bound for mutation
        up = 1  # upper bound for mutation
        indpb = 0.5  # mutation probability

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []

            for bad in self.worst:
                ret = np.random.normal(self.mu, self.sigma, (1, self.ind_size))
                ret = np.clip(ret, 0, 1)[0]
                new_pop += [ret]

            for ret in self.retain:
                ret = np.random.normal(self.mu, self.sigma, (1, self.ind_size))
                ret = np.clip(ret, 0, 1)[0]
                new_pop += [ret]

            ind_list = self.new_population(np.array(new_pop), ind_init) + self.new_population(np.array(self.best), ind_init)

        return ind_list


class UMDA_Adv(AdvEDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring,
                 adv_method_type, treatment_bad, treatment_ret_pred1, treatment_ret_pred0,
                 trick_network, extra_layers, early_stopping, **kwargs):

        super().__init__(eval_function, pop_sz, num_variables, method, offspring,
                         adv_method_type, treatment_bad, treatment_ret_pred1, treatment_ret_pred0,
                         trick_network, extra_layers, early_stopping, **kwargs)
        self.mu = None
        self.sigma = None
        self.best = None

        # print("It arrived here")
        # pass

    def update(self, population):
        X_train, y_train = self.select_best_worst(population, self.sel_method)
        self.mu = np.mean(self.best, axis=0)
        self.sigma = np.sqrt(np.var(self.best, axis=0))

        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs.

        if self.early_stopping == 1:
            prop_validation = 0.2
            n_train = len(X_train)
            n_val = int(prop_validation * n_train)
            rnd_perm = np.random.permutation(n_train)
            val_data = X_train[rnd_perm[:n_val], :]
            val_labels = y_train[rnd_perm[:n_val], :]
            train_data = X_train[rnd_perm[n_val:], :]
            train_labels = y_train[rnd_perm[:n_val], :]

            validation_data = (val_data, val_labels)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
            self.model.fit(train_data, train_labels, batch_size=10, epochs=250, verbose=0, callbacks=[callback],
                           validation_data=validation_data, shuffle=True, class_weight=None, sample_weight=None)
        elif self.early_stopping == 0:
            self.model.fit(X_train, y_train, batch_size=10, epochs=250, verbose=0, callbacks=[],
                           validation_data=[], shuffle=True, class_weight=None, sample_weight=None)

        # print("Whether network is tricked or not", self.trick_network)
        if self.trick_network == 1:
            n_layers = len(self.model.layers)
            last_layer_weights = self.model.layers[n_layers - 1].get_weights()[0]
            last_layer_biases = self.model.layers[n_layers - 1].get_weights()[1]
            new_weights = np.zeros((last_layer_weights.shape))
            new_biases = last_layer_biases  # np.zeros((last_layer_biases.shape))
            trials = 10
            npred = len(self.best)
            # print("Weights n-2",self.model.layers[n_layers-2].get_weights())
            # print("Weights n-1",self.model.layers[n_layers-1].get_weights())
            i = 0

            while (i < trials and npred > len(self.best) - 1):
                alpha = 0.01 * i
                new_weights[:, 0] = last_layer_weights[:, 0] * (1 + alpha)
                new_weights[:, 1] = last_layer_weights[:, 1] * (1 - alpha)
                # print(new_weights)
                self.model.layers[n_layers - 1].set_weights([new_weights, new_biases])
                npred = 0
                for b in self.best:
                    pred = np.argmax(self.model.predict(np.array([b])), axis=1)[0]
                    npred += pred
                # nretain = 0
                # for b in self.retain:
                #    pred = np.argmax(self.model.predict(np.array([b])), axis=1)[0]
                #    nretain += pred
                # print('pred ',i,npred,nretain)
                i = i + 1
        self.foolbox_model = fb.models.TensorFlowModel.from_keras(self.model, bounds=(0, 1))

    def generate(self, ind_init):
        eta = 0.1  # strength of polynomial mutation
        low = 0  # lower bound for mutation
        up = 1  # upper bound for mutation
        indpb = 0.5  # mutation probability

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []

            for bad in self.worst:
                pred = np.argmax(self.model.predict(np.array([bad])), axis=1)[0]
                # print(bad,"pred ",pred)
                if self.treatment_bad == 0:
                    bad = np.array(create_individual(len(bad)))
                elif self.treatment_bad == 1:
                    pass  # We do nothing, the solution is perturbed by AdvModel
                new_pop += [self.generate_adv_example(self.foolbox_model, bad, 0)]

            for ret in self.retain:
                pred = np.argmax(self.model.predict(np.array([ret])), axis=1)[0]

                if pred == 1:
                    ret = np.random.normal(self.mu, self.sigma, (1, self.ind_size))
                    ret = np.clip(ret, 0, 1)[0]

                elif pred == 0:
                    ret = self.generate_adv_example(self.foolbox_model, ret, 0)
                # print("pred-ret ",pred,len(ret),ret)
                new_pop += [ret]

            # print("Length is ", len(new_pop), " and total ", len(ind_init))
            # ind_list = self.new_population(np.array(new_pop), ind_init)
            ind_list = self.new_population(np.array(new_pop), ind_init) + self.new_population(np.array(self.best), ind_init)

        return ind_list