from matplotlib import colors, cm

class CMapp():
    def __init__(self, cmap, vmin, vmax, norm_base=colors.Normalize):
        self.norm = norm_base(vmin, vmax)
        self.cmap = cmap
        self.mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def __call__(self, value):
        return self.mappable.to_rgba(value)