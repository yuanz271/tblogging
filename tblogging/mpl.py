import numpy as np

__all__ = ["fig2tensor"]


def fig2tensor(fig):
    """
    Convert matplotlib figure to a tensor
    :param fig: figure object
    :return: image tensor
    """
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = np.expand_dims(
        img.reshape(fig.canvas.get_width_height()[::-1] + (3,)), axis=0
    )
    return img
