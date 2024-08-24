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

    #bad_sampler_low = BadSampler(eps= .0001, num_sample_points = 100)
    #bad_sampler_med = BadSampler(eps= 1, num_sample_points = 100)
    #bad_sampler_high = BadSampler(eps= 10000, num_sample_points = 100)

    #eg_sampler = ExpectedGradientsSampler(reference_dataset = validation_dataset, num_samples_per_line = 10, num_reference_points = 100)
    #eg_sampler = BallExpectedGradientsSampler(reference_dataset = validation_dataset, num_samples_per_line = 10, num_reference_points = 100, eps=1000)



    #ig = IntegratedGradients()
    #gv = GradientVariance()

    # igExplainer_low = GeneralizedIGExplainer(sampler = bad_sampler_low, measure = ig)
    # igExplainer_med = GeneralizedIGExplainer(sampler = bad_sampler_med, measure = ig)
    # igExplainer_high = GeneralizedIGExplainer(sampler = bad_sampler_high, measure = ig)

    #igExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = ig)

    #gvExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = gv)


    APExp = AttributionPriorExplainer(
        background_dataset = validation_dataset,
        batch_size = 2,
        k = 1,
        random_alpha = True,
        # scale_by_inputs = True
    )

    ig_visualizer = IGVisualizer(
        polarity = 'positive', 
        upper_quantile = 0.99, 
        lower_quantile = 0.0, 
        postive_color = [0, 255, 0],
        negative_color = [255, 0, 0], 
        overlay = True, 
        mask_mode = True, 
        overlay_weight = 0.3
    )

    ig_visualizer_v2 = IGVisualizer(
        polarity = 'positive', 
        upper_quantile = 0.99, 
        lower_quantile = 0.0, 
        postive_color = [0, 255, 0],
        negative_color = [255, 0, 0], 
        overlay = True, 
        mask_mode = False, 
        overlay_weight = 0.3
    )

    # gig_visualizer_0 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_1 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_2 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = True, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_3 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_4 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_5 = GIGVisualizer(
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = True, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_6 = GIGVisualizer(
    #     keep_magnitude =  True,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_7 = GIGVisualizer(
    #     keep_magnitude =  True,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_8 = GIGVisualizer(
    #     keep_magnitude =  True,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_9 = GIGVisualizer(
    #     keep_magnitude =  True,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # #clipped to quantiles
    # lq = 0.1
    # uq = 0.9


    # gig_visualizer_quant_0 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_1 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_2 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = False,
    #     scale_by_input = True, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_3 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_4 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_5 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  False,
    #     keep_sign = True,
    #     scale_by_input = True, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_6 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  True,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_7 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  True,
    #     keep_sign = False,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_8 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  True,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = False, 
    #     overlay_weight = 0.3
    # )

    # gig_visualizer_quant_9 = GIGVisualizer(
    #     lower_quantile= lq,
    #     upper_quantile= uq,
    #     keep_magnitude =  True,
    #     keep_sign = True,
    #     scale_by_input = False, 
    #     overlay = True, 
    #     overlay_weight = 0.3
    # )


    for current_batch_idx, batch_data in enumerate(validation_dataloader):
        if current_batch_idx < 5000:
            #goose is 5000
            #dog is 10000
            #crowd with religious figure is 20000
            continue

        # Extract the image and labels from batch
        images, labels = batch_data

        # Get the model output for this batch
        outputs = model(images)

        # Calculate the gradient measure
        # grad_measure_low = igExplainer_low.generate_explanation(images, model)
        # grad_measure_med = igExplainer_med.generate_explanation(images, model)
        # grad_measure_high = igExplainer_high.generate_explanation(images, model)
        #grad_measure_eg = igExplainer_egSampler.generate_explanation(images, model)

        # expected_gradients = APExp.shap_values(model, images)

        break
    # viz_image = images[0].detach().numpy()
    # viz_image = np.transpose(viz_image, (1,2,0))
    # viz_image = (viz_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    # viz_image = viz_image * 255

    # expected_gradient_image = expected_gradients[0].detach().numpy()
    # expected_gradient_image = np.transpose(expected_gradient_image, (1,2,0))

    # viz_grads_low = grad_measure_low[0].detach().numpy()
    # viz_grads_med = grad_measure_med[0].detach().numpy()
    # viz_grads_high = grad_measure_high[0].detach().numpy()
    #viz_grads_eg = grad_measure_eg[0].detach().numpy()
    
    #viz_grads_eg = np.moveaxis(viz_grads_eg, 0, -1)

    #print(viz_image.shape)
    #print(viz_grads_eg.shape)
    # print("viz_grads_eg: ", np.max(viz_grads_eg), np.min(viz_grads_eg))
    # plt.close() 
    # ig_histogram_r = plt.hist(viz_grads_eg[:,:,0], bins=50)
    # plt.savefig('ig_histogram_r.png')
    # plt.close() 
    # ig_histogram_g = plt.hist(viz_grads_eg[:,:,1], bins=50)
    # plt.savefig('ig_histogram_g.png')
    # plt.close() 
    # ig_histogram_b = plt.hist(viz_grads_eg[:,:,2], bins=50)
    # plt.savefig('ig_histogram_b.png')
    # plt.close() 
    # ig_histogram_r_range = plt.hist(viz_grads_eg[:,:,0], bins=50, range=(-.00001,.00001))
    # plt.savefig('ig_histogram_r_range.png')
    # plt.close() 
    # ig_histogram_g_range = plt.hist(viz_grads_eg[:,:,1], bins=50, range=(-.00001,.00001))
    # plt.savefig('ig_histogram_g_range.png')
    # plt.close() 
    # ig_histogram_b_range = plt.hist(viz_grads_eg[:,:,2], bins=50, range=(-.00001,.00001))
    # plt.savefig('ig_histogram_b_range.png')
    # plt.close() 

    # plt.close()
    # ig_image = ig_visualizer.visualize(viz_grads_eg, viz_image)
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('ig_image_mask.png')

    # plt.close() 
    # ig_image = ig_visualizer_v2.visualize(viz_grads_eg, viz_image)
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('ig_image_overlay.png')

    # plt.close() 
    # ig_image = gig_visualizer_0.visualize(viz_grads_eg, viz_image)
    # print("gig0: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig0_image.png')
    
    # plt.close() 
    # ig_image = gig_visualizer_1.visualize(viz_grads_eg, viz_image)
    # print("gig1: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig1_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_2.visualize(viz_grads_eg, viz_image)
    # print("gig2: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig2_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_3.visualize(viz_grads_eg, viz_image)
    # print("gig3: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig3_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_4.visualize(viz_grads_eg, viz_image)
    # print("gig4: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig4_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_5.visualize(viz_grads_eg, viz_image)
    # print("gig5: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig5_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_6.visualize(viz_grads_eg, viz_image)
    # print("gig6: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig6_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_7.visualize(viz_grads_eg, viz_image)
    # print("gig7: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig7_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_8.visualize(viz_grads_eg, viz_image)
    # print("gig8: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig8_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_9.visualize(viz_grads_eg, viz_image)
    # print("gig9: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig9_image.png')

    # #quantile clip
    # ig_image = gig_visualizer_quant_0.visualize(viz_grads_eg, viz_image)
    # print("gig0_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig0_quant_image.png')
    
    # plt.close() 
    # ig_image = gig_visualizer_quant_1.visualize(viz_grads_eg, viz_image)
    # print("gig1_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig1_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_2.visualize(viz_grads_eg, viz_image)
    # print("gig2_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig2_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_3.visualize(viz_grads_eg, viz_image)
    # print("gig3_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig3_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_4.visualize(viz_grads_eg, viz_image)
    # print("gig4_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig4_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_5.visualize(viz_grads_eg, viz_image)
    # print("gig5_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig5_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_6.visualize(viz_grads_eg, viz_image)
    # print("gig6_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig6_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_7.visualize(viz_grads_eg, viz_image)
    # print("gig7_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig7_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_8.visualize(viz_grads_eg, viz_image)
    # print("gig8_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig8_quant_image.png')

    # plt.close() 
    # ig_image = gig_visualizer_quant_9.visualize(viz_grads_eg, viz_image)
    # print("gig9_quant: ", np.max(ig_image), np.min(ig_image))
    # plt.imshow(ig_image.astype(np.uint8))
    # plt.savefig('gig9_quant_image.png')


    # plt.close()
    # plt.imshow(viz_image.astype(np.uint8))
    # plt.savefig('original_image.png')
    # plt.close()





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



    # Epsilon Trials with subplots


    ig = IntegratedGradients()
    gv = GradientVariance()
    #stab = Stability()
    #const = Consistency()
    stab_color = Stability_Color()
    const_color = Consistency_Color()



    k=5

    input_image = images[0]


    viz_image = input_image.detach().numpy()
    viz_image = np.transpose(viz_image, (1,2,0))
    viz_image = (viz_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    viz_image = viz_image * 255

    #attribution_trials = np.zeros(tuple([k,8]) + viz_image.shape)
    attribution_trials = np.zeros(tuple([3,8]) + viz_image.shape)
    print("Attribution Trials: ", attribution_trials.shape)
    #for i, epsilon in enumerate(range(1, 2000, 2000//k)):
    for i, epsilon in enumerate([1, 500, 2000]):

        print("Epsilon: ", epsilon)

        #create sampler and explainer
        eg_sampler = BallExpectedGradientsSampler(reference_dataset = validation_dataset, num_samples_per_line = 5, num_reference_points = 100, eps=epsilon)
        igExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = ig)
        gvExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = gv)
        #stabExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = stab)
        #constExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = const)
        stab_colorExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = stab_color)
        const_colorExplainer_egSampler = GeneralizedIGExplainer(sampler = eg_sampler, measure = const_color)
        
        #compute measure (IG)
        grad_measure_eg = igExplainer_egSampler.generate_explanation(images, model).squeeze(0).detach().numpy()
        print("Grad_Measure_eg: ", grad_measure_eg.shape)
        #viz_grads_eg = grad_measure_eg
        viz_grads_eg = np.moveaxis(grad_measure_eg, 0, -1)
        print("Viz_Grads_EG: ", viz_grads_eg.shape)
        

        #compute measure (GV)
        grad_measure_gv = gvExplainer_egSampler.generate_explanation(images, model).squeeze(0).detach().numpy()
        print("Grad_Measure_gv: ", grad_measure_gv.shape)
        viz_grads_gv = np.moveaxis(grad_measure_gv, 0, -1)
        print("Viz_Grads_GV: ", viz_grads_gv.shape)
        
        '''
        #compute measure (Stability)
        grad_measure_stab = stabExplainer_egSampler.generate_explanation(images, model).repeat(3,1,1).detach().numpy()
        #(expanded single-channel stability attribution back to 3 channels)
        print("Grad_Measure_stab: ", grad_measure_stab.shape)
        viz_grads_stab = np.moveaxis(grad_measure_stab, 0, -1)
        print("Viz_Grads_Stab: ", viz_grads_stab.shape)


        #compute measure (Consistency)
        grad_measure_const = constExplainer_egSampler.generate_explanation(images, model).repeat(3,1,1).detach().numpy()
        #(expanded single-channel consistency attribution back to 3 channels)
        print("Grad_Measure_const: ", grad_measure_const.shape)
        viz_grads_const = np.moveaxis(grad_measure_const, 0, -1)
        print("Viz_Grads_Const: ", viz_grads_const.shape)
        '''

        #compute measure (Stability in Color)
        grad_measure_stab_color = stab_colorExplainer_egSampler.generate_explanation(images, model).detach().numpy()
        #(expanded single-channel stability attribution back to 3 channels)
        print("Grad_Measure_stab_color: ", grad_measure_stab_color.shape)
        viz_grads_stab_color = np.moveaxis(grad_measure_stab_color, 0, -1)
        print("Viz_Grads_Stab_Color: ", viz_grads_stab_color.shape)


        #compute measure (Consistency in Color)
        grad_measure_const_color = const_colorExplainer_egSampler.generate_explanation(images, model).detach().numpy()
        #(expanded single-channel consistency attribution back to 3 channels)
        print("Grad_Measure_const_color: ", grad_measure_const_color.shape)
        viz_grads_const_color = np.moveaxis(grad_measure_const_color, 0, -1)
        print("Viz_Grads_Const_Color: ", viz_grads_const_color.shape)





        attribution_trials[i, 0] = viz_image

        #visualize using absolute and signed methods

        attribution_trials[i, 1] = gig_visualizer_unsigned.visualize(viz_grads_eg, viz_image)



    


        #unsigned_red = np.copy(attribution_trials[i, 1])
        #unsigned_red[:,:,(1,2)] = 0
        #unsigned_red = np.repeat(unsigned_red[:,:,0,np.newaxis], 3, axis=2)
        #unsigned_green = np.copy(attribution_trials[i, 1])
        #unsigned_green[:,:,(0,2)] = 0
        #unsigned_green = np.repeat(unsigned_green[:,:,1,np.newaxis], 3, axis=2)
        #unsigned_blue = np.copy(attribution_trials[i, 1])
        #unsigned_blue[:,:,(0,1)] = 0
        #unsigned_blue = np.repeat(unsigned_blue[:,:,2,np.newaxis], 3, axis=2)

        #attribution_trials[i, 2] = unsigned_red
        #attribution_trials[i, 3] = unsigned_green
        #attribution_trials[i, 4] = unsigned_blue

        attribution_trials[i, 2] = gig_visualizer_signed.visualize(viz_grads_eg, viz_image)

        #signed_red = np.copy(attribution_trials[i, 5])
        #signed_red[:,:,(1,2)] = 127
        #signed_red = np.repeat(signed_red[:,:,0,np.newaxis], 3, axis=2)
        #signed_green = np.copy(attribution_trials[i, 5])
        #signed_green[:,:,(0,2)] = 127
        #signed_green = np.repeat(signed_green[:,:,1,np.newaxis], 3, axis=2)
        #signed_blue = np.copy(attribution_trials[i, 5])
        #signed_blue[:,:,(0,1)] = 127
        #signed_blue = np.repeat(signed_blue[:,:,2,np.newaxis], 3, axis=2)

        #attribution_trials[i, 6] = signed_red
        #attribution_trials[i, 7] = signed_green
        #attribution_trials[i, 8] = signed_blue

        #attribution_trials[i, 9] = gig_visualizer_unsigned_scaled.visualize(viz_grads_eg, viz_image)
        #attribution_trials[i, 10] = gig_visualizer_signed_scaled.visualize(viz_grads_eg, viz_image)

        #attribution_trials[i, 11] = viz_image


        #visualize GV measure
        attribution_trials[i, 3] = gig_visualizer_unsigned.visualize(viz_grads_gv, viz_image)

        #visualize Stab measure
        #attribution_trials[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_stab, viz_image)
        #attribution_trials[i, 5] = gig_visualizer_signed.visualize(viz_grads_stab, viz_image)
        attribution_trials[i, 4] = gig_visualizer_unsigned.visualize(viz_grads_stab_color, viz_image)
        attribution_trials[i, 5] = gig_visualizer_signed.visualize(viz_grads_stab_color, viz_image)

        #visualize Const measure
        #attribution_trials[i, 6] = gig_visualizer_unsigned.visualize(viz_grads_const, viz_image)
        #attribution_trials[i, 7] = gig_visualizer_signed.visualize(viz_grads_const, viz_image)
        attribution_trials[i, 6] = gig_visualizer_unsigned.visualize(viz_grads_const_color, viz_image)
        attribution_trials[i, 7] = gig_visualizer_signed.visualize(viz_grads_const_color, viz_image)






    print("Attribution Trials (Filled): ", attribution_trials.shape)
    create_subplot(attribution_trials, "epsilon_trials_3_row")



    plt.close()
    plt.imshow(viz_image.astype(np.uint8))
    plt.savefig('original_image.png')
    plt.close()
 