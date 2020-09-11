import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import rocCurve
import torch
import torchvision
import fasttext
from data import HatefulMemesDataset
import wandb
from tqdm import tqdm
from config import hparams
wandb.init(project="meme")

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_text_transform():
    ft_path = "temp.txt"
    with open(ft_path, "w") as ft:
        for line in open(hparams.get("train_path")):
            line = json.loads(line.strip())["text"]
            ft.write(line + '\n')

        language_transform = fasttext.train_unsupervised(
            str(ft_path),
            model=hparams.get("fasttext_model", "cbow"),
            dim=hparams.get("embedding_dim")
        )
    return language_transform


def _build_image_transform():
    image_dim = 224
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(image_dim, image_dim)
            ),
            torchvision.transforms.ToTensor(),
            # all torchvision models expect the same
            # normalization mean and std
            # https://pytorch.org/docs/stable/torchvision/models.html
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return image_transform


def _build_dataset(dataset_key, image_transform, text_transform):
    return HatefulMemesDataset(
        data_path=hparams.get(dataset_key, dataset_key),
        img_dir=hparams.get("img_dir"),
        image_transform=image_transform,
        text_transform=text_transform,
        # limit training samples only
        dev_limit=(
            hparams.get("dev_limit", None)
            if "train" in str(dataset_key) else None
        ),
        balance=True if "train" in str(dataset_key) else False,
    )


image_transform = _build_image_transform()
text_transform = _build_text_transform()
train_dataset = _build_dataset("train_path", image_transform=image_transform, text_transform=text_transform)
dev_dataset = _build_dataset("dev_path", image_transform, text_transform)
test_dataset = _build_dataset("test_path", image_transform, text_transform)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=4,
    num_workers=4
)

val_dataloader = torch.utils.data.DataLoader(
    dev_dataset,
    shuffle=False,
    batch_size=4,
    num_workers=4
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=4,
    num_workers=4
)

from models import LanguageAndVisionConcat
from data import make_submission_frame

# easiest way to get features rather than
# classification is to overwrite last layer
# with an identity transformation, we'll reduce
# dimension using a Linear layer, resnet is 2048 out
vision_module = torchvision.models.resnet152(
    pretrained=True
)
vision_module.fc = torch.nn.Linear(
    in_features=2048,
    out_features=hparams.get("vision_feature_dim")
)

language_module = torch.nn.Linear(
    in_features=hparams.get("embedding_dim"),
    out_features=hparams.get("language_feature_dim")
)

run_dict = dict(
    num_classes=hparams.get("num_classes", 2),
    loss_fn=torch.nn.CrossEntropyLoss(),
    language_module=language_module,
    vision_module=vision_module,
    language_feature_dim=hparams.get("language_feature_dim"),
    vision_feature_dim=hparams.get("vision_feature_dim"),
    fusion_output_size=hparams.get(
        "fusion_output_size", 512
    ),
    dropout_p=hparams.get("dropout_p", 0.1),
)

model = LanguageAndVisionConcat(**run_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

from utils import AverageMeter


def training_step(model, batch):
    model = model.train()
    preds, loss = model(
        text=batch["text"].to(device),
        image=batch["image"].to(device),
        label=batch["label"].to(device)
    )

    return {"loss": loss}


def validation_step(model, batch):
    model = model.eval()
    with torch.no_grad():
        preds, loss = model(
            text=batch["text"].to(device),
            image=batch["image"].to(device),
            label=batch["label"].to(device)
        )

    return {"batch_val_loss": loss}


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hparams.get("lr", 0.001)
)
schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

Nepochs = 10
train_loss, valid_loss = [], []
train_meter = AverageMeter('Train Loss', ':.4e')
val_meter = AverageMeter('Loss', ':.4e')
if not os.path.exists("Output"):
    os.mkdir("Output")
modelPath = 'Output/backbone-{}.pth'

for epoch in range(Nepochs):
    comment = f'Epoch: {epoch}'
    EPOCH_DIR = modelPath.format(Nepochs - 1)

    if os.path.exists(EPOCH_DIR):
        model = torch.load(EPOCH_DIR)
        print("Modal loaded: ", EPOCH_DIR)
        break

    idx = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        out = training_step(model, batch)
        loss = out['loss']
        loss.backward()
        optimizer.step()
        child_comment = 'Train Loss:{:.2f}'.format(loss.item())
        idx += 1
        train_meter.update(loss.item())

    for batch in val_dataloader:
        out = validation_step(model, batch)
        val_meter.update(out['batch_val_loss'].item())

    train_loss.append(train_meter.avg)
    valid_loss.append(val_meter.avg)
    wandb.log({"train_loss": train_meter.avg})
    wandb.log({"val_loss": val_meter.avg})
    # save the model
    torch.save(model, modelPath.format(epoch))

print("Reading submission file......")
submission = make_submission_frame(model, test_dataloader)
submission.to_csv("Output/submission.csv", index=False)
print(submission.groupby("label").proba.mean())
print(submission.label.value_counts())
print("Submission file done......")

val_frame_pred = make_submission_frame(model, val_dataloader)
y_test = dev_dataset.samples_frame.label.to_numpy()
y_probas = val_frame_pred.proba.to_numpy()
fig, ax = plt.subplots()
ax, auroc = rocCurve(y_test, y_probas, ax)
plt.savefig("roc.png")
# ROC
wandb.log({'roc': wandb.plots.ROC(y_test, np.asarray([1 - y_probas, y_probas]).T),
           'auroc': auroc})

