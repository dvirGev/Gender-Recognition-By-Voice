import matplotlib.pyplot as plt
import numpy as np



def plotPCA(explained_variance_ratio_):
    # % matplotlib inline
    # import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 123, step=1)
    y = np.cumsum(explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 706, step=1)) #change from 0-based array index to 1-based   human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')

    plt.show()

def indealcomp(explained_variance_ratio):
    y =  np.cumsum(explained_variance_ratio)
    for x in range(len(y)):
        if y[x] > 0.999:
            print(f'~~~We choose to take the feachers between 0 to {x}~~~')
            return x
    return len(y)