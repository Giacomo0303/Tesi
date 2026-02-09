from abc import abstractmethod, ABC
from timm import create_model, data


class BaseDataset(ABC):
    def __init__(self, root_path, img_size, batch_size, mean_std, model_name, seed):
        self.root_path = root_path
        self.classes = []
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

        if mean_std == "imagenet":
            if model_name is not None:
                model = create_model(model_name, pretrained=False)
                data_config = data.resolve_model_data_config(model)
                self.mean, self.std = data_config["mean"], data_config["std"]
                del model, data_config
            else:
                raise AttributeError("Specificare il nome del modello da utilizzare")
        else:
            self.mean, self.std = mean_std

    @abstractmethod
    def get_transform(self, train=True):
        pass

    @abstractmethod
    def get_train_loader(self, num_workers):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_search_loader(self, n_per_classes):
        pass
