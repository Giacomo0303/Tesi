import torch
from Datasets.Imagenet import ImageNet
from utils.FineTuneUtils import eval_loop

from utils.PruneUtils import count_params_no_mask

first_model_path = "C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\NAS\\Results_imagenet_deit\\deit_small_imagenet_first.pth"
second_model_path = "C:\\Users\\cvip\\Desktop\\Tesi_Lombardo\\src\\NAS\\Results_imagenet_deit\\deit_small_imagenet_first_iter4.pth"
model_name = "deit_small_patch16_224"
batch_size = 64

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_model = torch.load(first_model_path, weights_only=False).to(device)
    second_model = torch.load(second_model_path, weights_only=False).to(device)
    first_model.eval()
    second_model.eval()

    dataset = ImageNet(root_path="D:\\Lombardo\\ImageNet", batch_size=batch_size, model_name=model_name,
                       train_size=0.97)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    first_model_params = count_params_no_mask(first_model)
    second_model_params = count_params_no_mask(second_model)

    print("First model params: ", first_model_params)
    print("First model validation accuracy:")
    eval_loop(first_model, val_loader, torch.nn.CrossEntropyLoss(), device, dataset.classes, report=False)
    print("First model test accuracy:")
    eval_loop(first_model, test_loader, torch.nn.CrossEntropyLoss(), device, dataset.classes, report=False)

    first_model.to("cpu")
    second_model.to(device)
    print("-----------------------------")
    print("Second model params:", second_model_params)
    print("Second model validation accuracy:")
    eval_loop(second_model, val_loader, torch.nn.CrossEntropyLoss(), device, dataset.classes, report=False)
    print("Second model test accuracy:")
    eval_loop(second_model, test_loader, torch.nn.CrossEntropyLoss(), device, dataset.classes, report=False)


if __name__ == "__main__":
    main()