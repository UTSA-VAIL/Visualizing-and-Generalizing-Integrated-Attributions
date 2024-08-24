import numpy as np
import torch
import torch.nn.functional as F
from datamodule import MNISTDataModule
from models import MnistLinearModel
import matplotlib.pyplot as plt

def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    delta_X = (pre_processing(inputs, cuda) - pre_processing(baseline, cuda)).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0))
    integrated_grad = delta_X * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                                baseline=255.0 *np.random.random(inputs.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
    # do the pre-processing
    predict_idx = None
    gradients = []
    for input in inputs:
        input = pre_processing(input, cuda)
        output = model(input)
        output = F.softmax(output, dim=1)
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        if cuda:
            index = index.cuda()
        output = output.gather(1, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients, target_label_idx


# This needs to be 
def pre_processing(obs, cuda):
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor

if __name__ == '__main__':
    dm = MNISTDataModule(data_dir='/data/progressive_data_dropout/mnist', num_workers = 32)
    model = MnistLinearModel()

    # checkpoint_file = '/data/integrated_gradients/mnist/checkpoints/epoch=0-step=214.ckpt'
    
    # # Load in the model
    # model = model.load_from_checkpoint(checkpoint_file)

    # Switch model to evaluate mode
    model.eval()

    # Set the data module into fit mode
    dm.setup(stage='fit')

    train_loader = dm.train_dataloader()

    for images, labels in train_loader:
        single_image = images[0].detach().numpy()

        # print(images.grad)
        gradients, label_index = calculate_outputs_and_gradients([single_image], model, None, False)

        print(gradients.shape)
        
         # calculae the integrated gradients 
        attributions = random_baseline_integrated_gradients(single_image, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=50, num_random_trials=10, cuda = False)

        print(attributions.shape)

        fig, ax = plt.subplots()
        ax.imshow(attributions)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig('test.png')

        break