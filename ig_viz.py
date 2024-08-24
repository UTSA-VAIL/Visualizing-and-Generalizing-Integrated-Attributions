import numpy as np 
import warnings
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        print("Visualizer Class")
    
    def visualize(self, attributions):
        visualization = None
        return visualization

class IGVisualizer:
    def __init__(self, polarity = 'positive', upper_quantile = 1.0, lower_quantile = 0.0, postive_color = [0, 255, 0],
    negative_color = [255, 0, 0], overlay = False, mask_mode = False, overlay_weight = 0.5):
        Visualizer.__init__(self)
        self.polarity = polarity
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.postive_color = postive_color
        self.negative_color = negative_color
        self.overlay = overlay
        self.mask_mode = mask_mode
        self.overlay_weight = overlay_weight

    def combine_image_with_overlay(self, overlay, image):
        return self.overlay_weight * overlay + (1-self.overlay_weight) * image

    def polarity_clip(self, attributions):
        if self.polarity == 'positive':
            return np.clip(attributions, 0, 1)
        elif self.polarity == 'negative':
            return np.clip(attributions, -1, 0)
        else:
            raise NotImplementedError

    def scale_to_quantiles(self, attributions):
        upper_bound = np.quantile(attributions, self.upper_quantile)
        lower_bound = np.quantile(attributions, self.lower_quantile)

        # Clip attributions
        transformed_attributions = np.clip(attributions, lower_bound, upper_bound) / (upper_bound - lower_bound)

        return transformed_attributions

    # This function DOES NOT work in batches
    def visualize(self, attributions, input_image):
        # Apply polarity function to attribution
        attributions = self.polarity_clip(attributions)

        # Convert attribution to gray-scale
        attributions = np.average(attributions, axis = 2)

        # Clip and scale to desired attributions quantiles
        attributions = self.scale_to_quantiles(attributions)

        # Expand grayscale back to color
        attributions = np.expand_dims(attributions, 2)

        if self.overlay:
            if self.mask_mode:
                # Scale input image by attributions
                attributions = attributions * input_image
                
                # # Reverse color channels
                # attributions = attributions[:, :, (2, 1, 0)]
            else:
                # Change the gray scale to the desired color 
                attributions = attributions * self.postive_color

                attributions = self.combine_image_with_overlay(attributions, input_image)
        else:
            # Change the gray scale to the desired color 
            attributions = attributions * self.postive_color
            
        # Reverse color channels for plotting purposes
        # attributions = attributions[:, :, (2, 1, 0)]

        return attributions



class GIGVisualizer:
    def __init__(self, upper_quantile = 1.0, lower_quantile = 0.0, keep_magnitude =  False, keep_sign = True, scale_by_input = False, overlay = False, overlay_weight = 0.5):
        Visualizer.__init__(self)
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.keep_magnitude = keep_magnitude
        self.keep_sign = keep_sign
        self.scale_by_input = scale_by_input
        self.overlay = overlay
        self.overlay_weight = overlay_weight

        if keep_magnitude and scale_by_input:
            warnings.warn("Warning: argument scale_by_input is unused if argument keep_magnitude=True")


    def combine_image_with_overlay(self, overlay, image):
        return self.overlay_weight * overlay + (1-self.overlay_weight) * image

    def clip_to_quantiles(self, attributions):

        upper_bound = np.quantile(np.abs(attributions), self.upper_quantile)
        lower_bound = np.quantile(np.abs(attributions), self.lower_quantile)

        # clip both positive and negative gradients to same absolute quantile
        positive_attributions = np.where(attributions > 0 , attributions, 0)
        negative_attributions = np.where(attributions < 0 , attributions, 0)
        
        clipped_positive = np.clip(positive_attributions, lower_bound, upper_bound)
        clipped_negative = np.clip(negative_attributions, -upper_bound, -lower_bound)

        #combine clipped positive and negative attributions so that attributions are clipped in absolute value
        clipped_attributions = clipped_positive + clipped_negative

        return clipped_attributions


    # This function DOES NOT work in batches
    def visualize(self, attributions, input_image):

        # Clip attributions
        attributions = self.clip_to_quantiles(attributions)

        if self.keep_sign:
            #create gray image to use as backdrop
            blank_slate = np.ones(attributions.shape)*127.5

            if self.keep_magnitude:
                #draw on blank slate (gray) with attributions (darker or lighter)
                visualization = blank_slate + attributions

            else:
                #scale the attributions to [-1,1]
                noramlized_attributions = attributions/np.max(np.abs(attributions))

                if self.scale_by_input:
                    # Scale input image by attributions (need to scale to [-127.5, 127.5] so that we can add to blank slate)
                    relative_brightness = noramlized_attributions * input_image * 0.5
                    
                else:
                    #scale the attributions to [-127.5,127.5]
                    relative_brightness = noramlized_attributions*127.5


                #draw on blank slate (gray) with relative attributions (darker or lighter)
                visualization = blank_slate + relative_brightness


        else:
            #make all attributions positive
            abs_attributions = np.abs(attributions)

            if self.keep_magnitude:
                #return the absolute value of the attributions
                visualization = abs_attributions

            else:
                #scale the attributions to [0,1]
                noramlized_abs_attributions = abs_attributions/np.max(abs_attributions)

                if self.scale_by_input:
                    # Scale attributions by input
                    scaled_attributions = noramlized_abs_attributions * input_image
                    
                    visualization = scaled_attributions

                else:
                    #scale the attributions to [0,255]
                    visualization = noramlized_abs_attributions*255
                 
        
        if self.overlay and not self.scale_by_input:
            #splice input image with visualizations
            visualization = self.combine_image_with_overlay(visualization, input_image)


        #make sure visualization is an integer image
        visualization = visualization.astype(np.uint8)

        #clip attributions to [0,255]
        #visualization = np.clip(visualization, 0,255)

        return visualization
