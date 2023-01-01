import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


from model import GarbageModel
from util import pil_loader, prepare_image


root = r"C:\Users\LUKE_SARGEN\projects\classifier\data\test"
rows = 3
cols = 4
scale = 2.5

model_path = "logs/subset/version_49/last.ckpt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : {}".format(device))

m = GarbageModel.load_from_checkpoint(model_path, map_location=device)
m.eval()
m.freeze()
input_size = 256  # m.hparams.input_size, 128, 144, 160, 192, 224, 256, 288, 320, 384, 448
centercrop = False
classes = m.hparams.classes
model = m.model.to(device)

sample_ds = ImageFolder(root=root)

idxs = np.random.choice(range(len(sample_ds)), cols*rows, replace=False)
fig = plt.figure(figsize=(scale*cols, scale*rows))
for i in range(1, cols*rows+1):
    # Load and classify the image
    img_path, y = sample_ds.samples[idxs[i-1]]
    print(i, img_path, classes[int(y)])
    img_color = pil_loader(img_path)
    img = prepare_image(img_color, input_size, centercrop).to(device)
    ylogits = model(img)
    yclass = torch.softmax(ylogits, dim=1)
    yprob, yhat = torch.max(yclass, dim=1)

    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(img_color)
    color = "g" if sample_ds.classes[int(y)]==classes[int(yhat)] else "r"
    title_str = "{} ({:.02f} % {})".format(sample_ds.classes[int(y)], float(100*yprob), classes[int(yhat)])
    ax.set_title(title_str, color=color)


fig.tight_layout()
plt.show()
