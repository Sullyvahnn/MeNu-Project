import matplotlib.pyplot as plt
from IPython import display
import matplotlib
plt.ion()

def plot(scores, mean_scores):
    matplotlib.use('Qt5Agg')
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Generations")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show()


