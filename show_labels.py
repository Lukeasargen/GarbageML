import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


from model import GarbageModel
from util import pil_loader, prepare_image


root = "data/subset"
rows = 3
cols = 4
scale = 2.5

model_path = "lightning_logs/version_7/checkpoints/epoch=199-step=11799.ckpt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : {}".format(device))

m = GarbageModel.load_from_checkpoint(model_path, map_location=device)
m.eval()
m.freeze()
input_size = m.hparams.input_size
classes = m.hparams.classes
model = m.model.to(device)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop(input_size),
    T.ToTensor(),
    normalize
])

sample_ds = ImageFolder(root=root)

idxs = np.random.choice(range(len(sample_ds)), cols*rows, replace=False)
fig = plt.figure(figsize=(scale*cols, scale*rows))
for i in range(1, cols*rows+1):
    # Load and classify the image
    img_path, y = sample_ds.samples[idxs[i-1]]
    print(i, img_path, classes[int(y)])
    img_color = pil_loader(img_path)
    img = prepare_image(img_color, input_size)
    ylogits = model(img.to(device))
    yclass = torch.softmax(ylogits, dim=1)
    yprob, yhat = torch.max(yclass, dim=1)

    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(img_color)
    color = "g" if sample_ds.classes[int(y)]==classes[int(yhat)] else "r"
    title_str = "{} ({:.02f} % {})".format(sample_ds.classes[int(y)], float(100*yprob), classes[int(yhat)])
    ax.set_title(title_str, color=color)


fig.tight_layout()
plt.show()
