class Iden():
    def __init__(self):
        pass

    def __getitem__(self, x):
        return x
    
    def __setitem__(self, x):
        raise TypeError("Identity objects do not support assignment")
    
class NullIterable():
    def __init__(self):
        pass

    def append(self, item):
        pass

    def __iter__(self):
        yield