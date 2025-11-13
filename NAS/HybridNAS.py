import torch, timm
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from NAS_Utils import find_target_emb, find_target_QK, find_target_V_proj, find_target_head, find_target_mlp

class HybridNAS:
    def __init__(self, model, loss_fn, search_loader, device):
        self.base_model = model
        self.loss_fn = loss_fn
        self.device = device
        self.dataloader = search_loader
        self.best_value = float("inf")
        self.best_state = None
        self.actions = [find_target_QK, find_target_V_proj, find_target_head, find_target_mlp, find_target_emb]

    def build_initial_state(self) -> None:
        start_state = {}
        start_state["embed_pruned_dims"] = []
        start_state["blocks"] = []

        n_blocks = len(self.base_model.blocks)

        for block in range(n_blocks):
            block_state = {
                "head_pruned_idx": [],
                "qk_pruned_dims": [],
                "v_proj_pruned_dims": [],
                "mlp_pruned_dims": []
            }
            start_state["blocks"].append(block_state)

        start_state["obj_val"] = float("inf")

    def bound(self, state):
        pass

    def branch(self, state) -> list[dict]:
        pass

    def apply_pruning(self, state):
        pass

    def eval_model(self, model):
        pass

    def search(self):
        self.build_initial_state()
        stack = [self.start_state]

        while len(stack) > 0:
            current_state = stack.pop()
            model = self.apply_pruning(current_state)
            self.eval_model(model)
            next_states = self.branch(current_state)

            for state in next_states:
                if not(self.bound(state)):
                    stack.append(state)

# TESTING

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 204
model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes).to(device)
checkpoint = torch.load("D:\\Tesi\\FirstFineTuning\\best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

data_config = timm.data.resolve_model_data_config(model)
imagenet_mean, imagenet_std = data_config["mean"], data_config["std"]

search_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

path = "D:\\Tesi\\Sets\\Set1\\search"
batch_size = 128

search_set = ImageFolder(root=path, transform=search_transform)
search_loader = torch.utils.data.DataLoader(search_set, batch_size=batch_size, shuffle=False, num_workers=1)

nas = HybridNAS(model, loss_fn=nn.CrossEntropyLoss(), search_loader=search_loader, device=device)
nas.build_initial_state()
