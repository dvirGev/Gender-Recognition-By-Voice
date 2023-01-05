import matplotlib.pyplot as plt
import numpy as np



def plotPCA(explained_variance_ratio_):
    # % matplotlib inline
    # import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 641, step=1)
    y = np.cumsum(explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 641, step=1)) #change from 0-based array index to 1-based   human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')

    plt.show()


def find_closest_to_zero(derivatives):
  min_diff = 0.00001 #good for 104
  min_index = None
  for i, d in enumerate(derivatives):
    diff = abs(d)
    if diff < min_diff:
        # min_diff = diff
        min_index = i
        break
  return derivatives[min_index], min_index

def indealcomp(explained_variance_ratio):
    derivatives = []
    for i in range(1, len(np.cumsum(explained_variance_ratio))):
        y2 = np.cumsum(explained_variance_ratio)[i]
        y1 = np.cumsum(explained_variance_ratio)[i - 1]
        x2 = i
        x1 = i - 1
        derivative = (y2 - y1) / (x2 - x1)
        derivatives.append(derivative)

    # print("derivatives", derivatives)
    return find_closest_to_zero(derivatives)[1]