import matplotlib.pyplot as plt

def set_sizing(small=17, medium=19, bigger=21):
    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=small)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=bigger)  # fontsize of the figure title