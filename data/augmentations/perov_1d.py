class Normalize1D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        t = (sample - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return t


def normalize(mean, std):

    t = Normalize1D(mean, std)

    return t
