import numpy as np
from sklearn.cluster import KMeans


def create_segments(spec_obs, temp_back, n_limit=5, n_samp=50000):
    x_node = spec_obs[:, 0]
    y_node = spec_obs[:, 1] - temp_back
    y_node[y_node < 0.] = 0.
    y_node /= np.trapz(y_node, x_node)
    d_node = np.diff(y_node)
    d_node = np.append(d_node[0], d_node)

    samps = Sampler(x_node, y_node).sample(n_samp)
    segments = segregate(samps, x_node, d_node, n_limit)
    return segments


def segregate(samps, x_node, d_node, n_limit):
    segments = []
    queue = [samps]
    while len(queue) > 0:
        samps = queue.pop(0)
        labels = KMeans(2).fit_predict(samps[:, None])
        x_min = samps.min()
        x_max = samps.max()
        cond = (x_node > x_min) & (x_node < x_max)

        samps_1 = samps[labels == 0]
        samps_2 = samps[labels == 1]
        if _check(samps_1, x_node, d_node, n_limit) \
            and _check(samps_1, x_node, d_node, n_limit):
            queue.append(samps_1)
            queue.append(samps_2)
        else:
            segments.append(cond)
    return segments


def _check(samps, x_node, d_node, n_limit):
    x_min = samps.min()
    x_max = samps.max()
    cut = np.mean(samps)
    cond = (x_node > x_min) & (x_node < x_max)
    x_node = x_node[cond]
    d_node = d_node[cond]
    d_left = d_node[x_node < cut]
    d_right = d_node[x_node >= cut]

    if np.count_nonzero(d_left > 0) < n_limit:
        return False
    if np.count_nonzero(d_right < 0) < n_limit:
        return False
    return True


class Sampler:
    def __init__(self, x_node, y_node):
        # x_node (N,)
        # y_node (N,)
        width = np.diff(x_node) # (N - 1,)
        c_node = np.zeros_like(x_node) # (N)
        c_node[1:] = np.cumsum(.5*(y_node[:-1] + y_node[1:])*width)

        self.x_node = x_node
        self.y_node = y_node
        self.c_node = c_node
        self.width = width

    def cdf(self, x_val):
        inds = np.searchsorted(self.x_node, x_val) - 1
        inds[inds < 0] = 0
        inds[inds >= len(self.x_node) - 1] = len(self.x_node) - 2

        width = self.width[inds]
        y_0 = self.y_node[inds]
        y_1 = self.y_node[inds + 1]
        x_local = (x_val - self.x_node[inds])/width
        cdf = width*(.5*x_local*x_local*(y_1 - y_0) + x_local*y_0) + self.c_node[inds]
        cdf /= self.c_node[-1]
        return cdf

    def sample(self, n_samp):
        c_val = np.random.rand(n_samp)
        c_node = self.c_node/self.c_node[-1]
        inds = np.searchsorted(c_node, c_val) - 1
        inds[inds < 0] = 0
        inds[inds >= len(self.x_node) - 1] = len(self.x_node) - 2

        width = self.width[inds]
        y_0 = self.y_node[inds]
        y_1 = self.y_node[inds + 1]
        v_diff = y_1 - y_0
        delta = y_0*y_0 + 2.*v_diff*(c_val - c_node[inds])/width
        x_local = (np.sqrt(delta) - y_0)/v_diff
        samps = self.x_node[inds] + width*x_local
        return samps


class SegmentLoss:
    def __init__(self, spec_obs, temp_back, n_limit, n_samp):
        self._segments = create_segments(spec_obs, temp_back, n_limit, n_samp)

    def __call__(self, y_pred, y_obs):
        loss_list = []
        for inds in self._segments:
            loss_list.append(np.mean(np.abs(y_pred[inds] - y_obs[inds])))
        loss_list.sort()
        return np.mean(loss_list[:5])
