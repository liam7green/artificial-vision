class LossHistory:
    def __init__(self):
        self.losses = []
        
    def on_train_begin(self):
        self.losses = []
        
    def on_batch_end(self, loss):
        self.losses.append(loss.item())

class ValLossHistory:
    def __init__(self):
        self.losses = []
        
    def on_train_begin(self):
        self.losses = []
        
    def on_batch_end(self, loss):
        self.losses.append(loss.item())