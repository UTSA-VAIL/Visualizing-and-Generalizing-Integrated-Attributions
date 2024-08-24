from re import L
from numpy import identity
import torch

class Measure:
    def __init__(self):
        print("Measure Class")
    
    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        measure_value = None
        return measure_value


class IntegratedGradients(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute the mean along the sample points
        measure_value = torch.mean(sample_gradients, dim = 1)
        return measure_value


class GradientVariance(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute square of gradients (E[grads^2] - E[grads]^2)
        squared_grads = torch.pow(sample_gradients, 2)
        expected_squared_grads = torch.mean(squared_grads, dim = 1)
        
        expected_grads = torch.mean(sample_gradients, dim = 1)
        squared_expected_grads = torch.pow(expected_grads, 2)

        # Compute the difference of the expectation of squares and the squared expectation
        measure_value = expected_squared_grads - squared_expected_grads
        return measure_value


class Stability(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute E[theta], where theta is the angle between (sample-input) and gradients
        # Currently only set up to work for a single input, not a batch

        #print("Input: ", input.shape)
        num_sample_points = sample.shape[1]

        repeated_input = input.repeat(num_sample_points, 1, 1, 1)
        #print("Input Repeated: ", repeated_input.shape)

        offset = repeated_input.unsqueeze(0)-sample
        #print("Offset: ", offset.shape)

        #print("Grads: ", sample_gradients.shape)

        #compute cosine similarity along color channel dimension to retain per-pixel attributions
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cos(offset, sample_gradients)
        #print("Similarity: ", similarity.shape)

        # # Compute the mean along the sample points
        measure_value = torch.mean(similarity, dim = 1)
        return measure_value


class Consistency(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute E[theta], where theta is the angle between gradients(sample) and gradients(input)
        # Currently only set up to work for a single input, not a batch

        print("Input Grads: ", input_gradients.shape)

        print("Input: ", input.shape)
        num_sample_points = sample_gradients.shape[1]

        repeated_input_grads = input_gradients.repeat(1, num_sample_points, 1, 1, 1)
        print("Input Repeated: ", repeated_input_grads.shape)

        print("Grads: ", sample_gradients.shape)

        #compute cosine similarity along color channel dimension to retain per-pixel attributions
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cos(repeated_input_grads, sample_gradients)
        print("Similarity: ", similarity.shape)

        # # Compute the mean along the sample points
        measure_value = torch.mean(similarity, dim = 1)
        return measure_value


#Color versions of stability and Consistency

class Stability_Color(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute E[theta], where theta is the angle between (sample-input) and gradients
        # Currently only set up to work for a single input, not a batch

        #print("Input: ", input.shape)
        num_sample_points = sample.shape[1]

        repeated_input = input.repeat(num_sample_points, 1, 1, 1)
        #print("Input Repeated: ", repeated_input.shape)

        offset = repeated_input.unsqueeze(0)-sample
        #print("Offset: ", offset.shape)

        #print("Grads: ", sample_gradients.shape)

        #compute cosine similarity between each pair of color channels to retain per-pixel attributions
        offset_r = offset[:,:,0,:,:]
        print("Offset Red: ", offset_r.shape)
        offset_g = offset[:,:,1,:,:]
        offset_b = offset[:,:,2,:,:]

        sample_gradients_r = sample_gradients[:,:,0,:,:]
        print("Grads Red: ", sample_gradients_r.shape)
        sample_gradients_g = sample_gradients[:,:,1,:,:]
        sample_gradients_b = sample_gradients[:,:,2,:,:]

        #create 3 2D vectors from pairs of color channels
        offset_rg = torch.stack([offset_r,offset_g], dim=2)
        print("Offset Red-Green: ", offset_rg.shape)
        offset_gb = torch.stack([offset_g,offset_b], dim=2)
        offset_br = torch.stack([offset_b,offset_r], dim=2)

        sample_gradients_rg = torch.stack([sample_gradients_r,sample_gradients_g], dim=2)
        print("Grads Red-Green: ", sample_gradients_rg.shape)
        sample_gradients_gb = torch.stack([sample_gradients_g,sample_gradients_b], dim=2)
        sample_gradients_br = torch.stack([sample_gradients_b,sample_gradients_r], dim=2)


        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity_rg = cos(offset_rg, sample_gradients_rg)
        print("Similarity Red-Green: ", similarity_rg.shape)
        similarity_gb = cos(offset_gb, sample_gradients_gb)
        similarity_br = cos(offset_br, sample_gradients_br)

        # # Compute the mean along the sample points
        measure_value_rg = torch.mean(similarity_rg, dim = 1)
        print("Measure Value Red-Green: ", similarity_rg.shape)
        measure_value_gb = torch.mean(similarity_gb, dim = 1)
        measure_value_br = torch.mean(similarity_br, dim = 1)

        #Restack the three measures to obtain a color image
        #measure_value = torch.stack([measure_value_rg, measure_value_gb, measure_value_br])        #not alt
        #measure_value = torch.stack([measure_value_br, measure_value_rg, measure_value_gb])        #alt_1
        measure_value = torch.stack([measure_value_gb, measure_value_br, measure_value_rg])         #alt_2      
        
        print("Measure Value: ", measure_value.shape)

        return measure_value


class Consistency_Color(Measure):
    def __init__(self):
        Measure.__init__(self)

    def compute_measure(self, sample_gradients, input, sample, input_gradients):
        # Compute E[theta], where theta is the angle between gradients(sample) and gradients(input)
        # Currently only set up to work for a single input, not a batch

        print("Input Grads: ", input_gradients.shape)

        print("Input: ", input.shape)
        num_sample_points = sample_gradients.shape[1]

        repeated_input_grads = input_gradients.repeat(1, num_sample_points, 1, 1, 1)
        print("Input Repeated: ", repeated_input_grads.shape)

        print("Grads: ", sample_gradients.shape)

        #compute cosine similarity between each pair of color channels to retain per-pixel attributions
        repeated_input_grads_r = repeated_input_grads[:,:,0,:,:]
        print("Input Repeated Red: ", repeated_input_grads_r.shape)
        repeated_input_grads_g = repeated_input_grads[:,:,1,:,:]
        repeated_input_grads_b = repeated_input_grads[:,:,2,:,:]

        sample_gradients_r = sample_gradients[:,:,0,:,:]
        print("Grads Red: ", sample_gradients_r.shape)
        sample_gradients_g = sample_gradients[:,:,1,:,:]
        sample_gradients_b = sample_gradients[:,:,2,:,:]

        #create 3 2D vectors from pairs of color channels
        repeated_input_grads_rg = torch.stack([repeated_input_grads_r,repeated_input_grads_g], dim=2)
        print("Input Repeated Red-Green: ", repeated_input_grads_rg.shape)
        repeated_input_grads_gb = torch.stack([repeated_input_grads_g,repeated_input_grads_b], dim=2)
        repeated_input_grads_br = torch.stack([repeated_input_grads_b,repeated_input_grads_r], dim=2)

        sample_gradients_rg = torch.stack([sample_gradients_r,sample_gradients_g], dim=2)
        print("Grads Red-Green: ", sample_gradients_rg.shape)
        sample_gradients_gb = torch.stack([sample_gradients_g,sample_gradients_b], dim=2)
        sample_gradients_br = torch.stack([sample_gradients_b,sample_gradients_r], dim=2)


        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity_rg = cos(repeated_input_grads_rg, sample_gradients_rg)
        print("Similarity Red-Green: ", similarity_rg.shape)
        similarity_gb = cos(repeated_input_grads_gb, sample_gradients_gb)
        similarity_br = cos(repeated_input_grads_br, sample_gradients_br)

        # # Compute the mean along the sample points
        measure_value_rg = torch.mean(similarity_rg, dim = 1)
        print("Measure Value Red-Green: ", similarity_rg.shape)
        measure_value_gb = torch.mean(similarity_gb, dim = 1)
        measure_value_br = torch.mean(similarity_br, dim = 1)

        #Restack the three measures to obtain a color image
        #measure_value = torch.stack([measure_value_rg, measure_value_gb, measure_value_br])        #not alt
        #measure_value = torch.stack([measure_value_br, measure_value_rg, measure_value_gb])         #alt_1
        measure_value = torch.stack([measure_value_gb, measure_value_br, measure_value_rg])         #alt_2


        print("Measure Value: ", measure_value.shape)

        return measure_value