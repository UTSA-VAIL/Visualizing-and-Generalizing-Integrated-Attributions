import functools
import operator
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from sampler import *
from measures import *
from torchvision import models

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GeneralizedIGExplainer(object):
    def __init__(self, sampler, measure):
        print('Explainer Class')
        self.sampler = sampler
        self.measure = measure

    def get_grads(self, sample_tensor, model):
        # (Batch size, num_sample_points, input_size)
        batch_size = sample_tensor.size(dim = 0)
        sample_size = sample_tensor.size(dim = 1)
        model_input_size = tuple([batch_size * sample_size]) + sample_tensor.size()[2:]

        # Allow gradient calculation
        sample_tensor.requires_grad = True

        # Resize and send through the model
        model_inputs = sample_tensor.view(model_input_size)
        sample_output = model(model_inputs)

        # Calculate the gradients
        model_grads = grad(
            outputs = sample_output,
            inputs = model_inputs,
            grad_outputs = torch.ones_like(sample_output),
            create_graph = True
        )

        # Reshape back to original size and return the gradients
        model_grads_reshape = model_grads[0].view(sample_tensor.size())
        return model_grads_reshape

    def generate_explanation(self, input_tensor, model):
        # Psudo Code
        # Step 1: Get sample points
        sample_points = self.sampler.generate_sample_points(input_tensor)
        print("Sample Points: ", sample_points.shape)

        # Step 2: Compute the gradients for every sample point using specified model
        grad_tensor = self.get_grads(sample_points, model)
        print("Grad Tensor: ", grad_tensor.shape)

        #       : Compute the gradients at the input point for use in the consistency measure
        input_grads = self.get_grads(input_tensor.unsqueeze(0), model)

        # Step 3: Compute some function f of gradients for EVERY sample point
        #       : Compute the mean of the f over the sample points *if desired
        #       : Also pass input, sample points, and input gradients for use in advanced measures
        grad_measure = self.measure.compute_measure(grad_tensor, input_tensor, sample_points, input_grads)
        # print(grad_measure.shape)
        

        return grad_measure

if __name__ == '__main__':
    bad_sampler = BadSampler(eps= 10, num_sample_points = 10)
    ig = IntegratedGradients()

    model = models.resnet18(pretrained = False, num_classes = 2)
    random_tensor = torch.rand(5, 3, 32, 32)

    igExplainer = GeneralizedIGExplainer(sampler = bad_sampler, measure = ig)

    igExplainer.awesome_method(random_tensor, model)