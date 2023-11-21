import matplotlib.pyplot as plt
import torch
import os
from Models.Pytorch.STIResNet import STIResNet

ACTUAL_PATH = os.getcwd()
IMG_PATH = '../../Imagenes/STI/'
PHASE_PATH = IMG_PATH + 'Phase/'
CHI_PATH = IMG_PATH + 'Chi/'
device = torch.device('cpu')
phase_img = 'phase_test.pt'
chi_img = 'chi_test.pt'

state_dict = torch.load('checkpoints/test20/state_dicts.pt', map_location=device)

epoch = state_dict['epoch']
print(epoch)
train_loss = state_dict['train_loss'][0:epoch]
val_loss = state_dict['val_loss'][0:epoch]

fig, axs = plt.subplots(1, 2)
fig.suptitle('Model Errors')
axs[0].plot(train_loss)
axs[0].set_title('Train Loss')
axs[0].set_xlabel('Epoch')

axs[1].plot(val_loss)
axs[1].set_title('Val Loss')
axs[1].set_xlabel('Epoch')

sti_model = STIResNet(in_channels=6, out_channels=6, init_features=16)
sti_model.load_state_dict(state_dict['model_state_dict'])

phase_test = torch.load(PHASE_PATH + phase_img).permute(-1, 0, 1, 2).unsqueeze(0).float()
chi_test = torch.load(CHI_PATH + chi_img).permute(-1, 0, 1, 2).unsqueeze(0).float()

n_tensor = 2

fig, axs = plt.subplots(1, 2)
axs[0].imshow(phase_test[:, n_tensor, :, :, 24].squeeze(), cmap='gray')
axs[0].set_title('Phase')
axs[1].imshow(chi_test[:, n_tensor, :, :, 24].squeeze().detach(), cmap='gray')
axs[1].set_title('Chi')

phase_test *= 1e6
chi_test *= 1e6

chi_model = sti_model(phase_test)

_, axs = plt.subplots(1, 3)
axs[0].imshow(phase_test[:, n_tensor, :, :, 24].squeeze(), cmap='gray')
axs[0].set_title('Phase input')
axs[1].imshow(chi_test[:, n_tensor, :, :, 24].squeeze(), cmap='gray')
axs[1].set_title('Ground Truth')
axs[2].imshow(chi_model[:, n_tensor, :, :, 24].squeeze().detach(), cmap='gray')
axs[2].set_title('Output')

n_tensor = 0

_, axs = plt.subplots(1, 3)
axs[0].imshow(phase_test[:, n_tensor, :, :, 24].squeeze(), cmap='gray')
axs[0].set_title('Phase input')
axs[1].imshow(chi_test[:, n_tensor, :, :, 24].squeeze(), cmap='gray')
axs[1].set_title('Ground Truth')
axs[2].imshow(chi_model[:, n_tensor, :, :, 24].squeeze().detach(), cmap='gray')
axs[2].set_title('Output')

plt.show()
