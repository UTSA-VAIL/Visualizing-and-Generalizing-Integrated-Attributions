import os
# Set the enviroment variable for pretrained docker purposes
os.environ['TORCH_HOME'] = 'pytorch_models/'

import numpy as np
from torchvision import *
from torch.utils.data import DataLoader
from datasets import *
from sampler import *
from measures import * 
from generalized_gradients import GeneralizedIGExplainer
#from shap2.shap.plots import image
import matplotlib.pyplot as plt
from attributionpriors.attributionpriors.pytorch_ops_cpu import AttributionPriorExplainer
from ig_viz import IGVisualizer, GIGVisualizer

from PIL import Image, ImageDraw, ImageFont




# Figures needs to be in grid shape
def create_subplot(figures, figure_name):
    figsize = (figures.shape[1], figures.shape[0])
    print("Figsize: ", figsize)
    dpi = 240
    fig, axs = plt.subplots(nrows=figures.shape[0], ncols=figures.shape[1], figsize=figsize, dpi=dpi)
    plt.subplots_adjust(
        left=None,
        bottom=None, 
        right=None,
        top=None, 
        wspace=0.01, 
        hspace=0.01
    )

    for row_id in range(figures.shape[0]):
        for col_id in range(figures.shape[1]):
            image = figures[row_id][col_id].astype(np.uint8)
            #image = np.transpose(image, (1,2,0))
            #image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            #image = image * 255
            #image = image.astype(np.uint8)
            if figures.shape[0]==1 and figures.shape[1]==1:
                axs.imshow(image)
                axs.axis('off')
            elif figures.shape[0]==1:
                axs[col_id].imshow(image)
                axs[col_id].axis('off')
            elif figures.shape[1]==1:
                axs[row_id].imshow(image)
                axs[row_id].axis('off')
            else:
                axs[row_id, col_id].imshow(image)
                axs[row_id, col_id].axis('off')

 
    plt.savefig(figure_name + '.png', bbox_inches='tight', pad_inches = 0.01)

if __name__ == '__main__':


    
    validation_dataset = ImagenetTest(image_dir = '/data/imagenet/val')
    num_classes = 1000
    model = models.resnet18(pretrained = True)

    validation_dataloader = DataLoader(
        dataset = validation_dataset, 
        batch_size = 1, 
        num_workers = 8, 
        shuffle = False
    )

    
    for current_batch_idx, batch_data in enumerate(validation_dataloader):

        #select a handful of images for use in figure


        if current_batch_idx == 5000:
            #goose is 5000
            #dog is 10000

            # Extract the image and labels from batch
            images_1, labels_1 = batch_data
            # Get the model output for this batch
            outputs_1 = model(images_1)


        if current_batch_idx == 10000:
            #goose is 5000
            #dog is 10000

            # Extract the image and labels from batch
            images_2, labels_2 = batch_data
            # Get the model output for this batch
            outputs_2 = model(images_2)


        if current_batch_idx == 20000:
            #goose is 5000
            #dog is 10000

            # Extract the image and labels from batch
            images_3, labels_3 = batch_data
            # Get the model output for this batch
            outputs_3 = model(images_3)

            break
    

    input_image_1 = images_1[0]
    viz_image_1 = input_image_1.detach().numpy()
    viz_image_1 = np.transpose(viz_image_1, (1,2,0))
    viz_image_1 = (viz_image_1 * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    viz_image_1 = viz_image_1 * 255

    input_image_2 = images_2[0]
    viz_image_2 = input_image_2.detach().numpy()
    viz_image_2 = np.transpose(viz_image_2, (1,2,0))
    viz_image_2 = (viz_image_2 * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    viz_image_2 = viz_image_2 * 255

    input_image_3 = images_3[0]
    viz_image_3 = input_image_3.detach().numpy()
    viz_image_3 = np.transpose(viz_image_3, (1,2,0))
    viz_image_3 = (viz_image_3 * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    viz_image_3 = viz_image_3 * 255



    
    #clipped to quantiles
    lq = 0.05
    uq = 0.95

    gig_visualizer_unsigned = GIGVisualizer(
        lower_quantile= lq,
        upper_quantile= uq,
        keep_magnitude =  False,
        keep_sign = False,
        scale_by_input = False, 
        overlay = False, 
        overlay_weight = 0.3
    )

    gig_visualizer_unsigned_scaled = GIGVisualizer(
        lower_quantile= lq,
        upper_quantile= uq,
        keep_magnitude =  False,
        keep_sign = False,
        scale_by_input = True, 
        overlay = False, 
        overlay_weight = 0.3
    )

    gig_visualizer_signed = GIGVisualizer(
        lower_quantile= lq,
        upper_quantile= uq,
        keep_magnitude =  False,
        keep_sign = True,
        scale_by_input = False, 
        overlay = False, 
        overlay_weight = 0.3
    )

    gig_visualizer_signed_scaled = GIGVisualizer(
        lower_quantile= lq,
        upper_quantile= uq,
        keep_magnitude =  False,
        keep_sign = True,
        scale_by_input = True, 
        overlay = False, 
        overlay_weight = 0.3
    )
    



    





    



    '''
    #G-IG Figures/local IG figures
    ig = IntegratedGradients()

    epsilons = [1, 500, 2000]

    #create array to hold all gig visualizations
    gig_figs = np.zeros(tuple([len(epsilons),6]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        igExplainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = ig)


        #compute measure (IG)
        #first input image
        grad_measure_eg_eps = igExplainer_egSampler_eps.generate_explanation(images_1, model).squeeze(0).detach().numpy()
        print("Grad_Measure_eg: ", grad_measure_eg_eps.shape)
        viz_grads_eg_eps = np.moveaxis(grad_measure_eg_eps, 0, -1)
        print("Viz_Grads_EG: ", viz_grads_eg_eps.shape)

        gig_figs[i, 0] = viz_image_1

        gig_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_eg_eps, viz_image_1)
        gig_figs[i, 2] = gig_visualizer_signed.visualize(viz_grads_eg_eps, viz_image_1)


        #second input image
        grad_measure_eg_eps = igExplainer_egSampler_eps.generate_explanation(images_2, model).squeeze(0).detach().numpy()
        print("Grad_Measure_eg: ", grad_measure_eg_eps.shape)
        viz_grads_eg_eps = np.moveaxis(grad_measure_eg_eps, 0, -1)
        print("Viz_Grads_EG: ", viz_grads_eg_eps.shape)

        gig_figs[i, 3] = viz_image_2

        gig_figs[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_eg_eps, viz_image_2)
        gig_figs[i, 5] = gig_visualizer_signed.visualize(viz_grads_eg_eps, viz_image_2)


    create_subplot(gig_figs, "measure_gig")
    '''

    '''
    #GV Figures
    gv = GradientVariance()

    epsilons = [1, 500, 2000]

    #create array to hold all visualizations
    gv_figs = np.zeros(tuple([len(epsilons),4]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        gvExplainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = gv)


        #compute measure (GV)
        #first input image
        grad_measure_gv_eps = gvExplainer_egSampler_eps.generate_explanation(images_1, model).squeeze(0).detach().numpy()
        print("Grad_Measure_gv: ", grad_measure_gv_eps.shape)
        viz_grads_gv_eps = np.moveaxis(grad_measure_gv_eps, 0, -1)
        print("Viz_Grads_GV: ", viz_grads_gv_eps.shape)

        gv_figs[i, 0] = viz_image_1

        gv_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_gv_eps, viz_image_1)


        #second input image
        grad_measure_gv_eps = gvExplainer_egSampler_eps.generate_explanation(images_2, model).squeeze(0).detach().numpy()
        print("Grad_Measure_gv: ", grad_measure_gv_eps.shape)
        viz_grads_gv_eps = np.moveaxis(grad_measure_gv_eps, 0, -1)
        print("Viz_Grads_GV: ", viz_grads_gv_eps.shape)

        gv_figs[i, 2] = viz_image_2

        gv_figs[i, 3] = gig_visualizer_unsigned.visualize(viz_grads_gv_eps, viz_image_2)


    create_subplot(gv_figs, "measure_gv")
    '''






    ''''
    #stab Figures
    stab = Stability()

    epsilons = [1, 500, 2000]

    #create array to hold all visualizations
    stab_figs = np.zeros(tuple([len(epsilons),6]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        stabExplainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = stab)


        #compute measure (STAB)
        #first input image
        grad_measure_stab_eps = stabExplainer_egSampler_eps.generate_explanation(images_1, model).repeat(3,1,1).detach().numpy()
        print("Grad_Measure_stab: ", grad_measure_stab_eps.shape)
        viz_grads_stab_eps = np.moveaxis(grad_measure_stab_eps, 0, -1)
        print("Viz_Grads_STAB: ", viz_grads_stab_eps.shape)

        stab_figs[i, 0] = viz_image_1

        stab_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_stab_eps, viz_image_1)
        stab_figs[i, 2] = gig_visualizer_signed.visualize(viz_grads_stab_eps, viz_image_1)


        #second input image
        grad_measure_stab_eps = stabExplainer_egSampler_eps.generate_explanation(images_2, model).repeat(3,1,1).detach().numpy()
        print("Grad_Measure_stab: ", grad_measure_stab_eps.shape)
        viz_grads_stab_eps = np.moveaxis(grad_measure_stab_eps, 0, -1)
        print("Viz_Grads_STAB: ", viz_grads_stab_eps.shape)

        stab_figs[i, 3] = viz_image_2

        stab_figs[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_stab_eps, viz_image_2)
        stab_figs[i, 5] = gig_visualizer_signed.visualize(viz_grads_stab_eps, viz_image_2)


    create_subplot(stab_figs, "measure_stab")
    '''



    '''
    #const Figures
    const = Consistency()

    epsilons = [1, 500, 2000]

    #create array to hold all visualizations
    const_figs = np.zeros(tuple([len(epsilons),6]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        constExplainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = const)


        #compute measure (CONST)
        #first input image
        grad_measure_const_eps = constExplainer_egSampler_eps.generate_explanation(images_1, model).repeat(3,1,1).detach().numpy()
        print("Grad_Measure_const: ", grad_measure_const_eps.shape)
        viz_grads_const_eps = np.moveaxis(grad_measure_const_eps, 0, -1)
        print("Viz_Grads_CONST: ", viz_grads_const_eps.shape)

        const_figs[i, 0] = viz_image_1

        const_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_const_eps, viz_image_1)
        const_figs[i, 2] = gig_visualizer_signed.visualize(viz_grads_const_eps, viz_image_1)


        #second input image
        grad_measure_const_eps = constExplainer_egSampler_eps.generate_explanation(images_2, model).repeat(3,1,1).detach().numpy()
        print("Grad_Measure_const: ", grad_measure_const_eps.shape)
        viz_grads_const_eps = np.moveaxis(grad_measure_const_eps, 0, -1)
        print("Viz_Grads_CONST: ", viz_grads_const_eps.shape)

        const_figs[i, 3] = viz_image_2

        const_figs[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_const_eps, viz_image_2)
        const_figs[i, 5] = gig_visualizer_signed.visualize(viz_grads_const_eps, viz_image_2)


    create_subplot(const_figs, "measure_const")
    '''

    
    #color stab Figures
    stab_color = Stability_Color()

    epsilons = [1, 500, 2000]

    #create array to hold all visualizations
    stab_color_figs = np.zeros(tuple([len(epsilons),6]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        stab_color_Explainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = stab_color)


        #compute measure (STAB COLOR)
        #first input image
        grad_measure_stab_color_eps = stab_color_Explainer_egSampler_eps.generate_explanation(images_1, model).detach().numpy()
        print("Grad_Measure_STAB_COLOR: ", grad_measure_stab_color_eps.shape)
        viz_grads_stab_color_eps = np.moveaxis(grad_measure_stab_color_eps, 0, -1)
        print("Viz_Grads_STAB_COLOR: ", viz_grads_stab_color_eps.shape)

        stab_color_figs[i, 0] = viz_image_1

        stab_color_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_stab_color_eps, viz_image_1)
        stab_color_figs[i, 2] = gig_visualizer_signed.visualize(viz_grads_stab_color_eps, viz_image_1)


        #second input image
        grad_measure_stab_color_eps = stab_color_Explainer_egSampler_eps.generate_explanation(images_2, model).detach().numpy()
        print("Grad_Measure_STAB_COLOR: ", grad_measure_stab_color_eps.shape)
        viz_grads_stab_color_eps = np.moveaxis(grad_measure_stab_color_eps, 0, -1)
        print("Viz_Grads_STAB_COLOR: ", viz_grads_stab_color_eps.shape)

        stab_color_figs[i, 3] = viz_image_2

        stab_color_figs[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_stab_color_eps, viz_image_2)
        stab_color_figs[i, 5] = gig_visualizer_signed.visualize(viz_grads_stab_color_eps, viz_image_2)


    create_subplot(stab_color_figs, "measure_stab_color_alt_2")
    



    
    #color const Figures
    const_color = Consistency_Color()

    epsilons = [1, 500, 2000]

    #create array to hold all visualizations
    const_color_figs = np.zeros(tuple([len(epsilons),6]) + viz_image_1.shape)

    for i, epsilon in enumerate(epsilons):

        print("Epsilon: ", epsilon)

        eg_sampler_eps = BallExpectedGradientsSampler(reference_dataset = validation_dataset, 
                                              num_samples_per_line = 5, 
                                              num_reference_points = 100, 
                                              eps=epsilon)
        const_color_Explainer_egSampler_eps = GeneralizedIGExplainer(sampler = eg_sampler_eps, measure = const_color)


        #compute measure (CONST COLOR)
        #first input image
        grad_measure_const_color_eps = const_color_Explainer_egSampler_eps.generate_explanation(images_1, model).detach().numpy()
        print("Grad_Measure_const_color: ", grad_measure_const_color_eps.shape)
        viz_grads_const_color_eps = np.moveaxis(grad_measure_const_color_eps, 0, -1)
        print("Viz_Grads_CONST_COLOR: ", viz_grads_const_color_eps.shape)

        const_color_figs[i, 0] = viz_image_1

        const_color_figs[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_const_color_eps, viz_image_1)
        const_color_figs[i, 2] = gig_visualizer_signed.visualize(viz_grads_const_color_eps, viz_image_1)


        #second input image
        grad_measure_const_color_eps = const_color_Explainer_egSampler_eps.generate_explanation(images_2, model).detach().numpy()
        print("Grad_Measure_const_color: ", grad_measure_const_color_eps.shape)
        viz_grads_const_color_eps = np.moveaxis(grad_measure_const_color_eps, 0, -1)
        print("Viz_Grads_CONST_COLOR: ", viz_grads_const_color_eps.shape)

        const_color_figs[i, 3] = viz_image_2

        const_color_figs[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_const_color_eps, viz_image_2)
        const_color_figs[i, 5] = gig_visualizer_signed.visualize(viz_grads_const_color_eps, viz_image_2)


    create_subplot(const_color_figs, "measure_const_color_alt_2")
    






