class HybridNAS:
    def __init__(self, model, loss_fn, search_loader, device):
        self.base_model = model
        self.loss_fn = loss_fn
        self.device = device
        self.dataloader = search_loader
        #self.actions = []

    def check_constraints(self):
        pass

    def next_states(self):
        pass

    def search(self):
        pass
