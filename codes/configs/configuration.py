unavailable_path = ["\\data\\sr_benchmark\\BSDS"]

class Configuration():
    def __init__(self):
        self.epochs = 1
        self.batch_size = 20
        self.model_name = "srgen_demo" + str(self.epochs) + ".pth"
        self.data_path = [
                "\\data\\sr_benchmark\\BSDS100",
                "\\data\\sr_benchmark\\BSDS200",
                "\\data\\sr_benchmark\\General100" ]

    def show(self):
        for path in self.data_path:
            print (path)
