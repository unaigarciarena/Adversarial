if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(50001), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    my_seed = args.integers[0]  # Seed: Used to set different outcomes of the stochastic program
    number_variables = args.integers[1]
    svr = args.integers[2]
    pop_size = args.integers[3]
    gens = args.integers[4]
    problems = [F1, F2, F3, F4, F5, F6, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24]
    p = args.integers[5]

    outs = args.integers[6]
    retain_ws = args.integers[7]
    retain_inds = args.integers[8]
    sol_disc = args.integers[9]
    train_m = args.integers[10]
    adv_method_type = args.integers[11]  # Type of adversarial method to use.
    # Between 0 and 15, applies one specific method
    # For type=16 selects randomly among the previous 15

    treatment_bad = args.integers[12]  # What to do with bad solutions?
    # 0: Use them as starting point of adversarial
    # 1: Substitute by random and use as starting point of adversarial

    treatment_ret_pred1 = args.integers[13]  # What to do with retain solutions predicted as good ?
    # 0: Apply random mutation and pass it to adversarial
    # 1: Select good solution, apply random mutation and pass to adv
    # 2: Generate random solution and pass to adversarial

    treatment_ret_pred0 = args.integers[14]  # What to do with retain solutions predicted as bad ?
    # 0: Apply random mutation and pass it to adversarial
    # 1: Select good solution, apply random mutation and pass to adv
    # 2: Generate random solution and pass to adversarial
    # 3: Pass it directly to adversarial

    trick_network = args.integers[15]  # Whether to trick the network biasing the weights from the last layer
    # to wrongly predict more labels as 1 as 0
    # 0: Nothing done to the network
    # 1: Network is tricked

    n_extra_layers = args.integers[16]  # Hidden layers added to increase complexity.
    # 0: No hidden layers added beyond the initial 2
    # i: 2*i hidden layers added

    early_stopping = args.integers[17]  # Early stopping for learning the neural network
    # 0: Not used
    # 1: It is used

    strt = time.time()
    for p in [0, 5, 10, 1, 6, 11]:
        for trick_network in [0, 1]:
            for my_seed in range(5):
                print(my_seed, p, trick_network, my_seed)
                problem = problems[p](number_variables)
                a, b = main(my_seed)
                print((time.time() - strt) / 3600, "hours")
                np.save("Evals_" + str(my_seed) + "_" + str(number_variables) + "_" + str(svr) + "_" + str(pop_size) + "_" + str(gens) + "_" + str(p) + "_" + str(outs) + "_" + str(retain_ws) + "_" + str(retain_inds) + "_" + str(sol_disc) + "_" + str(train_m) + "_" + str(adv_method_type) + "_" + str(
                    treatment_bad) + "_" + str(treatment_ret_pred1) + "_" + str(treatment_ret_pred0) + "_" + str(trick_network) + "_" + str(n_extra_layers) + "_" + str(early_stopping) + ".npy", np.array(evals))
