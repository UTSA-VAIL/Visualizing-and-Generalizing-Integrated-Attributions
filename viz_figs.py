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
            else:
                axs[row_id, col_id].imshow(image)
                axs[row_id, col_id].axis('off')

    
    if not(figures.shape[0]==1 and figures.shape[1]==1):
        
        col_labels = ['Input', 'Unsigned', 'Signed']
        row_labels = ['Unmultiplied', 'Multiplied']

        pad = 1 # in points

        csfont = {'fontname':'Times New Roman'}
        for ax, col in zip(axs[0], col_labels):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size=5, ha='center', va='baseline')

        for ax, row in zip(axs[:,0], row_labels):
            ax.annotate(row, xy=(0.5, 1), xytext=(-34, -40),
                        xycoords='axes fraction', textcoords='offset points',
                        size=5, ha='left', va='baseline', rotation = 90)



    plt.savefig(figure_name + '.png', bbox_inches='tight', pad_inches = 0.01)


if __name__ == '__main__':


    '''
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
        if current_batch_idx < 20000:
            #goose is 5000
            #dog is 10000
            continue

        # Extract the image and labels from batch
        images, labels = batch_data

        # Get the model output for this batch
        outputs = model(images)
    '''


    
    #clipped to quantiles
    lq = 0
    uq = 1

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
    ig = IntegratedGradients()
    gv = GradientVariance()
    stab = Stability()
    const = Consistency()
    

    k=5

    input_image = images[0]


    viz_image = input_image.detach().numpy()
    viz_image = np.transpose(viz_image, (1,2,0))
    viz_image = (viz_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    viz_image = viz_image * 255
    '''




    #load example attribution

    #im = Image.open("hopper.jpg")
    #a = np.asarray(im)


    
    test_attribtuion = Image.new('RGB', (224, 224))
    draw = ImageDraw.Draw(test_attribtuion)
    draw.ellipse((0, 75, 56, 131), fill=(255, 0, 0), outline=(127, 127, 127), width=3)
    draw.ellipse((56, 75, 112, 131), fill=(0, 255, 0), outline=(127, 127, 127), width=3)
    draw.ellipse((112, 75, 168, 131), fill=(0, 0, 255), outline=(127, 127, 127), width=3)
    draw.ellipse((168, 75, 224, 131), fill=(255, 255, 255), outline=(127, 127, 127), width=3)
    
    draw.rectangle((24, 24, 200, 34), fill=(127, 127, 127))
    draw.rectangle((24, 190, 200, 200), fill=(127, 127, 127))

    font = ImageFont.truetype("FreeMono.ttf", 15)
    text = 'ATTRIBUTION INFORMATION'

    draw.text((10, 4), font = font, text = text, align ='left') 
    draw.text((10, 204), font = font, text = text, align ='left') 



    draw.ellipse((24, 40, 200, 50), fill=(200, 0, 200))
    draw.ellipse((24, 174, 200, 184), fill=(0, 200, 200))


    test_attribtuion.save('test_attribution.jpg', quality=95)
    


    test_input = Image.new('RGB', (224, 224))
    draw_input = ImageDraw.Draw(test_input)

    # specified font size
    #font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20) 
    
    font = ImageFont.truetype("FreeMono.ttf", 20)
    text = 'INPUT FEATURES'

    draw_input.text((30, 92), font = font, text = text, align ='left',  fill=(0,255,255) ) 
    draw_input.text((30, 112), font = font, text = text, align ='left',  fill=(255,255,255) ) 
    draw_input.text((30, 132), font = font, text = text, align ='left',  fill=(255,0,255) ) 


    test_input.save('test_input.jpg', quality=95)



    #convert both images to numpy
    test_input_np = np.asarray(test_input)


    test_attribtuion_np = np.asarray(test_attribtuion, dtype=float)

    #make some areas of the attribution negative
    #test_attribtuion_np_neg = -1*test_attribtuion_np
    #test_attribtuion_np[:105] = test_attribtuion_np_neg[:105]

    test_attribtuion_np[:105,:112] *= -1
    test_attribtuion_np[106:,113:] *= -1


    #print("Min (neg): ", test_attribtuion_np_neg.min())
    print("Min (before viz): ", test_attribtuion_np.min())


    #print(test_input_np.shape)
    #print(test_attribtuion_np.shape)



    subfigs = test_visualization = np.zeros(tuple([2,3]) + test_attribtuion_np.shape)

    test_att_unsigned =  gig_visualizer_unsigned.visualize(test_attribtuion_np, test_input_np)
    test_att_unsigned = np.expand_dims(test_att_unsigned, axis=(0,1))
    print(test_att_unsigned.shape)
    create_subplot(test_att_unsigned, 'test_attribution_unsigned')

    test_att_signed =  gig_visualizer_signed.visualize(test_attribtuion_np, test_input_np)
    test_att_signed = np.expand_dims(test_att_signed, axis=(0,1))
    print(test_att_signed.shape)
    create_subplot(test_att_signed, 'test_attribution_signed')

    test_att_unsigned_scaled =  gig_visualizer_unsigned_scaled.visualize(test_attribtuion_np, test_input_np)
    test_att_unsigned_scaled = np.expand_dims(test_att_unsigned_scaled, axis=(0,1))
    print(test_att_unsigned_scaled.shape)
    create_subplot(test_att_unsigned_scaled, 'test_attribution_unsigned_scaled')

    test_att_signed_scaled =  gig_visualizer_signed_scaled.visualize(test_attribtuion_np, test_input_np)
    test_att_signed_scaled = np.expand_dims(test_att_signed_scaled, axis=(0,1))
    print(test_att_signed_scaled.shape)
    create_subplot(test_att_signed_scaled, 'test_attribution_signed_scaled')


    subfigs[0,0] = test_input_np
    subfigs[1,0] = test_input_np

    subfigs[0,1] = test_att_unsigned
    subfigs[0,2] = test_att_signed

    subfigs[1,1] = test_att_unsigned_scaled
    subfigs[1,2] = test_att_signed_scaled

    create_subplot(subfigs, 'visualization_comparison')


