class Config_Table(object):
    def __init__(self, args):
        self.n_classes = args.n_class
        self.source = args.source
        self.train = args.train
        self.test = args.test

        self.n_feature = 0
        self.class_dict = {"S": 0, "B": 1, "M": 2, "E": 3}
        self.invert_class_dict = {0: "S", 1: "B", 2: "M", 3: "E"}
