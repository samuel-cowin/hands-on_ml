from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

def plot_training_model(X, y, model, l=2):
    plot_decision_regions(X,y,clf=model,legend=l)
    plt.show()