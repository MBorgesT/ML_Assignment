import matplotlib.pyplot as plt

# plot errors
def plot_test_training_err(param_values, errors, title, hyperparam):
    errors = list(errors)
    test_err = [x[0] for x in errors]
    train_err = [x[1] for x in errors]

    plt.xlabel('$' + hyperparam + '$')
    plt.ylabel('$classification~~error$')
    plt.title(title)
    plt.plot(param_values,test_err,color='red')
    plt.plot(param_values,train_err,color='blue')
    plt.legend(['test','train'])
    plt.show()


# plot aucs
def plot_test_training_auc(param_values, aucs, title, hyperparam):
    auc_list = [x[0] for x in aucs]
    plt.xlabel('$' + hyperparam + '$')
    plt.ylabel('$AUC$')
    plt.title(title)
    plt.plot(param_values, auc_list, color='red')
    plt.show()


# plot auc tree num
def plot_tree_compare(ntree, auc_ada_tree, auc_gbc_tree, auc_tree, auc_decision_stump):
    plt.plot(ntree, [x for x in auc_ada_tree], color='red')
    plt.plot(ntree, [x for x in auc_gbc_tree], color='purple')
    plt.axhline([x for x in auc_tree], color='blue')
    plt.axhline([x for x in auc_decision_stump], color='black')
    plt.legend(['AdaBoost','GradientBoost','Tree','Stump'])
    plt.xlabel('Number of trees')
    plt.ylabel('ROC AUC')
    plt.title('Ensembles and trees in classification');
    plt.show()