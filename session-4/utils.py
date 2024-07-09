import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from IPython.display import display, HTML


def generate_data(
    l_mu, sig=None, n_samples=1000, class_ratio='balanced',
    random_state=None
):
    l_mu = np.array(l_mu)
    n_classes, n_features = l_mu.shape
    if class_ratio == 'balanced':
        class_ratio = np.ones(n_classes) / n_classes

    if sig is None:
        sig = np.eye(n_features)

    rng = check_random_state(random_state)
    X, y = [], []
    n_samples_generated = 0
    for i, (mu, r) in enumerate(zip(l_mu, class_ratio)):
        n_samples_class = int(n_samples * r)
        if i == n_classes - 1:
            n_samples_class = n_samples - n_samples_generated
        n_samples_generated += n_samples_class

        X.append(rng.multivariate_normal(
            mean=mu, cov=sig,
            size=n_samples_class,
            check_valid='raise'
        ))
        y.extend([i] * n_samples_class)

    X = np.vstack(X)
    y = np.array(y)

    return X, y


def plot_data(X, y, U=None, ax=None):

    if U is None:
        U = np.eye(2)
    assert U.shape[0] == 2

    if y.ndim == 1:
        y = OneHotEncoder().fit_transform(y[:, None]).toarray()

    n_classes = y.shape[1]
    class_colors = plt.get_cmap('viridis', n_classes)(range(n_classes))

    X = X @ U.T

    # Display the result in ax, which is created if it does not exist.
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c=y @ class_colors)


def show_decision_boundary(predict_proba=None, U=None, n_grid=250, ax=None,
                           lim=None, alpha=0.6, data=None):
    """Show decision boundary for predict proba
    """
    if U is None:
        U = np.eye(2)
    assert U.shape[0] == 2

    if lim is None:
        if data is not None:
            X = data[0] @ U
            lim = list(zip(X.min(axis=0), X.max(axis=0)))
        else:
            lim = ((-3, 3), (-3, 3))

    # Compute the probability of each class in the space
    x = np.linspace(*lim[0], n_grid)
    y = np.linspace(*lim[1], n_grid)
    XX, YY = np.meshgrid(x, y[::-1])
    coord = np.array([XX.flatten(), YY.flatten()]).T
    ZZ = predict_proba(X=coord @ U)

    # Compute colors associated to each class
    n_classes = ZZ.shape[1]
    class_colors = plt.get_cmap('viridis', n_classes)(range(n_classes))
    # set alpha to 0.3
    class_colors[:, -1] = alpha
    ZZ = ZZ @ class_colors

    # Display the result in ax, which is created if it does not exist.
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        ZZ.reshape(n_grid, n_grid, 4), aspect='auto',
        extent=(*lim[0], *lim[1])
    )

    # If data is provided, plot the scatter plot on top
    if data is not None:
        plot_data(*data, U=U, ax=ax)

    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])


def create_animation(l_predict_proba, X, y, iter_step, n_grid=250, alpha=0.6,
                     fname='anim.mp4'):

    n_iter = len(l_predict_proba)
    lim = list(zip(X.min(axis=0), X.max(axis=0)))

    if y.ndim == 1:
        y = OneHotEncoder().fit_transform(y[:, None]).toarray()

    n_classes = y.shape[1]
    class_colors = plt.get_cmap('viridis', n_classes)(range(n_classes))

    # First create the figure, and instantiate the element we want to animate.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], c=y @ class_colors)

    # Compute the probability of each class in the space
    # set alpha to 0.3
    class_colors[:, -1] = alpha
    x = np.linspace(*lim[0], n_grid)
    y = np.linspace(*lim[1], n_grid)
    XX, YY = np.meshgrid(x, y[::-1])
    coord = np.array([XX.flatten(), YY.flatten()]).T
    ZZ = l_predict_proba[0](X=coord) @ class_colors

    im = ax.imshow(
        ZZ.reshape(n_grid, n_grid, 4), aspect='auto',
        extent=(*lim[0], *lim[1])
    )

    # animation function. This is called sequentially.
    def animate(i):
        ZZ = l_predict_proba[i](X=coord) @ class_colors
        im.set_data(ZZ.reshape(n_grid, n_grid, 4))
        return (im,)

    # call the animator. blit=True means only re-draw the parts that have
    # changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, n_iter, iter_step),
        interval=50, blit=True
    )
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20)
    anim.save(fname, writer=writer)
    plt.close(fig)

    call = f"""
        <video width=50% controls>
            <source src="./{fname}" type="video/mp4">
        </video>
    """
    display(HTML(call))
