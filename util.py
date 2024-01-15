class Iden():
    def __init__(self):
        pass

    def __getitem__(self, x):
        return x
    
    def __setitem__(self, x):
        raise TypeError("Identity objects do not support assignment")