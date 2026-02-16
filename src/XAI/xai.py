import timm
from src.utils.NAS_Utils import load_model
import json
from src.utils.XAIutils import RWC, analize_mlp, analize_qk, analize_vproj, analize_head

model_name = "vit_small_patch16_224"
save_path = "D:\\Tesi\\src\\FineTuning"
report_path = "D:\\Tesi\\src\\NAS\\ResultsFinal\\Distil\\pruning_report_89.json"
num_classes = 100
img_size = 224
batch_size = 128
N_epochs = 30

original_model = timm.create_model(model_name=model_name, pretrained=True)

finetuned_model = load_model(model_name="vit_small_patch16_224", num_classes=num_classes,
                             path="D:\\Tesi\\src\\FineTuning\\best_model.pth")

pruning_report = json.load(open(report_path))

analize_mlp(original_model, finetuned_model, pruning_report)
analize_qk(original_model, finetuned_model, pruning_report)
analize_vproj(original_model, finetuned_model, pruning_report)
analize_head(original_model, finetuned_model, pruning_report)
