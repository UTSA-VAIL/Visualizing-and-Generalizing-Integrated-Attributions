import torch
from torch.utils.data import RandomSampler, DataLoader
from viz import create_subplot

# Base Class Framework
class Sampler:
    def __init__(self):
        print('Do nothing')

    def generate_sample_points(self, num_sample_points = 1):
        # TODO: Calculate sample points
        sample_points = None
        return sample_points

# Description: Generates samples within a ball of radius eps around the input (not uniform)
class BadSampler(Sampler):
    def __init__(self, eps = 0.01, num_sample_points = 1):
        self.eps = eps
        self.num_sample_points = num_sample_points
        Sampler.__init__(self)

    def generate_sample_points(self, input_tensor):
        batch_size = input_tensor.size(dim = 0)
        
        # Find a random tensor same size as input tensor
        # Reference direction shape: (Batch, num_samples, size of input tensor)
        reference_direction = torch.rand(size = tuple([batch_size,self.num_sample_points]) + input_tensor.size()[1:], dtype = torch.float32)
        normalized_reference_direction = torch.zeros(reference_direction.size())
        sample_points = torch.zeros(reference_direction.size())
        
        for batch_id in range(batch_size):
            for index in range(self.num_sample_points):
                reference_norm = torch.norm(reference_direction[batch_id, index], p = 2)
                normalized_reference_direction[batch_id, index] = reference_direction[batch_id,index] / reference_norm

                # Compute a random radius
                random_radius = torch.rand(1) * self.eps 

                # Get final scaled direction
                offset = normalized_reference_direction[batch_id, index] * random_radius

                # Get the final sample point
                sample_points[batch_id, index] = input_tensor[batch_id] + offset

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points

# RANDOMLY SPACING FOR LINE INTEGRALS 
class ExpectedGradientsSampler(Sampler):
    def __init__(self, reference_dataset, num_samples_per_line = 1, num_reference_points = 1):
        self.reference_dataset = reference_dataset
        self.num_samples_per_line = num_samples_per_line
        self.num_reference_points = num_reference_points

        Sampler.__init__(self)

    def generate_sample_points(self, input_tensor):
        batch_size = input_tensor.size(dim = 0)
        print('batch_size', batch_size)
        print('input_size', input_tensor.size()[1:])
        num_sample_points = self.num_reference_points * self.num_samples_per_line

        # Generate reference tensor
        reference_sampler = RandomSampler(
            data_source = self.reference_dataset, 
            replacement = True,
            num_samples = self.num_reference_points, 
        )

        reference_dataloader = DataLoader(
            dataset = self.reference_dataset,
            batch_size = self.num_reference_points,
            sampler = reference_sampler,
        )

        sample_points = torch.zeros(size = tuple([batch_size, num_sample_points]) + input_tensor.size()[1:])
        # reference_images = torch.zeros(size = tuple([batch_size, self.num_reference_points]) + input_tensor.size()[1:])
        print('Sample points', sample_points.shape)
        
        for batch_id in range(batch_size):
            reference_points = next(iter(reference_dataloader))[0].float()
            # print('Reference points', reference_points.shape)

            # reference_images[batch_id] = reference_points
            
            for reference_index in range(self.num_reference_points):
                for line_index in range(self.num_samples_per_line):

                    # Compute a random alpha
                    random_alpha = torch.rand(1)

                    # Get final scaled direction
                    line_sample = reference_points[reference_index] * random_alpha

                    # Get the final sample point
                    sample_points[batch_id, reference_index * self.num_samples_per_line + line_index] = input_tensor[batch_id] + line_sample

        # Plot reference images # TODO: Remove later
        # reference_images = reference_images.cpu().detach().numpy()
        # print('Reference images', reference_images.shape)
        # create_subplot(reference_images, 'reference_images')
        # sample_points_images = sample_points.cpu().detach().numpy()
        # create_subplot(sample_points_images, 'sample_points')

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points


# Expected Gradients within epsilon ball
class BallExpectedGradientsSampler(Sampler):
    def __init__(self, reference_dataset, num_samples_per_line = 1, num_reference_points = 1,  eps = 0.01):
        self.eps = eps
        self.reference_dataset = reference_dataset
        self.num_samples_per_line = num_samples_per_line
        self.num_reference_points = num_reference_points

        Sampler.__init__(self)

    def generate_sample_points(self, input_tensor):
        batch_size = input_tensor.size(dim = 0)
        print('batch_size', batch_size)
        print('input_size', input_tensor.size()[1:])
        num_sample_points = self.num_reference_points * self.num_samples_per_line

        # Generate reference tensor
        reference_sampler = RandomSampler(
            data_source = self.reference_dataset, 
            replacement = True,
            num_samples = self.num_reference_points, 
        )

        reference_dataloader = DataLoader(
            dataset = self.reference_dataset,
            batch_size = self.num_reference_points,
            sampler = reference_sampler,
        )

        sample_points = torch.zeros(size = tuple([batch_size, num_sample_points]) + input_tensor.size()[1:])
        #reference_images = torch.zeros(size = tuple([batch_size, self.num_reference_points]) + input_tensor.size()[1:])
        print('Sample points', sample_points.shape)
        
        for batch_id in range(batch_size):
            reference_points = next(iter(reference_dataloader))[0].float()
            # print('Reference points', reference_points.shape)

            #reference_images[batch_id] = reference_points
            
            for reference_index in range(self.num_reference_points):
                for line_index in range(self.num_samples_per_line):

                    # Scale reference direction to be of norm 1
                    reference_norm = torch.norm(reference_points[reference_index], p = 2)
                    normalized_reference_direction = reference_points[reference_index] / reference_norm

                    # Compute a random radius (equivalent to random radius for epsilon ball sampler)
                    random_alpha = torch.rand(1) * self.eps

                    # Get final scaled direction
                    scaled_line_sample = normalized_reference_direction * random_alpha

                    # Get the final sample point
                    sample_points[batch_id, reference_index * self.num_samples_per_line + line_index] = input_tensor[batch_id] + scaled_line_sample

        # Plot reference images # TODO: Remove later
        #reference_images = reference_images.cpu().detach().numpy()
        #print('Reference images', reference_images.shape)
        #create_subplot(reference_images, 'reference_images')
        #sample_points_images = sample_points.cpu().detach().numpy()
        #create_subplot(sample_points_images, 'sample_points_small')

        # Return Shape: (Batch, num_samples, size of input tensor)
        return sample_points




if __name__ == '__main__':
    bad_sampler = BadSampler(eps= 0.1, num_sample_points = 3)

    random_tensor = torch.rand(5, 2)

    sample_ponts = bad_sampler.generate_sample_points(random_tensor)

    print(sample_ponts.shape)