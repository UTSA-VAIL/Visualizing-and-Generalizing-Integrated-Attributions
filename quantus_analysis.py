import argparse
import logging
import os
import json
import torch

from pathlib import Path
from packages.utilities.logging_utilities import *
# Setup logging for other files to use
if os.path.exists('./logs/quantus.log'):
    os.remove('./logs/quantus.log')
addLoggingLevel('TRACE', logging.DEBUG - 5)
global_config_logger(log_file = './logs/quantus.log', log_level = logging.DEBUG)

from config import Config
from packages.utilities.eval_utilities import *

import torch
import gc
import quantus
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from captum.attr import *
from packages.utilities.general_utilities import UnNormalize



# TODO test_relevance_rank_accuracy (Localization --> not applicable)
# TODO max sensitivity (Robustness)
# TODO avg sensitivity (Robustness)
# TODO Pixel Flipping (Faithfulness, maybe not good for signed attributions)
# TODO Faithfulness Correlation (Faithfulness)
# TODO Monotonicity Metric (Faithfulness)
# TODO Relative Input Stability (RIS) (Robustness)
# TODO Sparseness (Complexity)




if __name__ == '__main__':
    # Setup the main logger
    logger = setup_logger(name = __name__)

    # Argument Parser init
    parser = argparse.ArgumentParser(description = 'Getting started tutorial for Quantus')

    parser.add_argument('--training_config_file_path', required = True, type = str, help = 'Get the path to the training config file.')
    #parser.add_argument('--explanation_config_file_path', required = False, type = str, default = None, help = 'Get the path to the explanation config file.')

    # Parse the arguments
    args = parser.parse_args()

    # Setup the config 
    c = Config(
        args.training_config_file_path,
        default_settings = './configs/default_training.json', 
        schema = './configs/training_schema.json',
        mode = 'test', 
    )

    # Set up the model dictionary 
    model_dict = setup_testing_dict(c)
    model_dict['model'].eval()


    # TESTING: MODEL EVAL TEST
    # # Run the evaluation
    # model_dict['testing_loss'] = evaluate_model(model_dict)

    # # Get metric calculation
    # model_dict['metric_df'] = calculate_testing_metrics(model_dict)

    # print(model_dict['metric_df'])
    # quit()


    # Set up the explanation config if provided
    # if args.explanation_config_file_path is not None:
    #     ec = Config(
    #         args.explanation_config_file_path, 
    #         default_settings = './configs/default_explainer.json',
    #         schema = './configs/explainer_schema.json',
    #         mode = 'test', 
    #     )
        
    #     measure_name = str(ec.config_dic['attribution']['measure'])
    #     radius = str(ec.config_dic['attribution']['sampler']['eps'])
    #     num_points_per_line = str(ec.config_dic['attribution']['sampler']['num_samples_per_line'])
    #     num_ref_points = str(ec.config_dic['attribution']['sampler']['num_reference_points'])

    #     measure_key = measure_name + '_' + radius + '_' + num_points_per_line + '_' + num_ref_points
        
    #     reference_dataset = ec.prepare_dataset(reference = True)
    #     explainer = ec.prepare_explainer(reference_dataset = reference_dataset)


    #method_list = ['IntegratedGradients', 'Saliency','GradientShap']
    method_list = ['FeatureAblation', 'FeaturePermutation','Deconvolution']

    measure_key = 'built-in'


    # exit if results file already exists
    metric_filename = 'logs/quantus_analysis/' + measure_key + '.csv'
    if Path(metric_filename).is_file():
        print("File ", metric_filename, " already exists. Exiting.")
        exit()

    # Create a dataloader
    explanation_dataloader = torch.utils.data.DataLoader(
        dataset = model_dict['testing_dataset'],
        shuffle = False,
        batch_size = 32,
        num_workers = 1,
        pin_memory = False
    )

    # Define the desired quantus metric

    metrics = []
    #metric_names = ['PixelFlipping', 'FaithfulnessCorrelation']
    metric_names = ['MaxSensitivity', 'AvgSensitivity']
    #metric_names = ['Sparseness', 'Complexity']

    metric_data = {}
    for metric_name in metric_names:
        for method in method_list:
            metric_key = metric_name + '_' + method
            metric_data[metric_key] = []



    # Faithfulness
    # pixel_flipping = quantus.PixelFlipping(
    #     features_in_step = 32,
    #     perturb_baseline = "uniform",
    #     perturb_func = quantus.perturb_func.baseline_replacement_by_indices,
    #     return_aggregate = True
    # )
    # metrics.append(pixel_flipping)

    # WARNING probably need to test with abs=true and abs=false
    # faithfulness_correlation = quantus.FaithfulnessCorrelation(
    #         nr_runs = 100,  
    #         subset_size = 32,  
    #         perturb_baseline = "uniform",
    #         perturb_func = quantus.perturb_func.baseline_replacement_by_indices,
    #         similarity_func = quantus.similarity_func.correlation_pearson,  
    #         abs = True,  
    #         return_aggregate = True, # was false until now
    #     )
    # metrics.append(faithfulness_correlation)

    # Return max sensitivity scores in an one-liner - by calling the metric instance.
    max_sensitivity = quantus.MaxSensitivity(
        nr_samples = 10,
        lower_bound = 0.2,
        norm_numerator = quantus.norm_func.fro_norm,
        norm_denominator = quantus.norm_func.fro_norm,
        perturb_func = quantus.perturb_func.uniform_noise,
        similarity_func = quantus.similarity_func.difference,
        return_aggregate = True
    )
    metrics.append(max_sensitivity)

    avg_sensitivity = quantus.AvgSensitivity(
        nr_samples = 10,
        lower_bound = 0.2,
        norm_numerator = quantus.norm_func.fro_norm,
        norm_denominator = quantus.norm_func.fro_norm,
        perturb_func = quantus.perturb_func.uniform_noise,
        similarity_func = quantus.similarity_func.difference,
        return_aggregate = True
    )
    metrics.append(avg_sensitivity)


    #Complexity
    # sparse = quantus.Sparseness(
    #         return_aggregate = True
    #     )
    # metrics.append(sparse)

    # complexity = quantus.Complexity(
    #         return_aggregate = True
    #     )
    # metrics.append(complexity)


    # Get a single batch
    for current_batch_idx, (images, labels) in enumerate(explanation_dataloader):

        # if current_batch_idx > 2:
        #     break

        print("Batch id:", current_batch_idx)

        # Send images and labels to the device
        images, labels = images.to(device = model_dict['device']), labels.to(device = model_dict['device'])

        # Get Attributions
        #print("Generating initial explanations...")
        # our_input_explanation = explainer.generate_explanation(
        #                             model_dict = model_dict, 
        #                             tensor_to_explain = images, 
        #                             input_sample_points = None, 
        #                             parameter_ref = None,
        #                         ).cpu().numpy()
        #print("Initial explanations successfully generated")

        images, labels = images.cpu().numpy(), labels.cpu().numpy()

        for i, metric in enumerate(metrics):

            print("Metric:", metric_names[i])

            metric_key = metric_names[i]# + '_' + measure_key

            print("Running Quantus metric...")
            # batch_results = metric(
            #             model = model_dict['model'], 
            #             x_batch = images, 
            #             y_batch = labels, 
            #             a_batch = our_input_explanation, 
            #             device = model_dict['device'],
            #             explain_func = explainer.quantus_explain,
            #             explain_func_kwargs = {"model_dict": model_dict}
            #         )
            batch_results = [metric(
                        model = model_dict['model'], 
                        x_batch = images, 
                        y_batch = labels, 
                        a_batch = None, 
                        device = model_dict['device'],
                        explain_func = quantus.explain,
                        explain_func_kwargs={"method": method}) for method in method_list
                    ]
            print("Quantus metric completed: ", batch_results)  
            
            for method_num, method in enumerate(method_list):
                metric_key = metric_names[i] + '_' + method
                metric_data[metric_key].append(batch_results[method_num])

        metric_data_pd = pd.DataFrame(metric_data)
        metric_data_pd.to_csv(metric_filename, index = False)     
        

    # Compute average for all batches
    for i, metric in enumerate(metrics):
        for method in method_list:
            metric_key = metric_names[i] + '_' + method
            metric_average = np.mean(metric_data[metric_key])

            average_key = metric_key + '_avg'
            metric_data[average_key] = metric_average

    metric_data_pd = pd.DataFrame(metric_data)
    metric_data_pd.to_csv(metric_filename, index = False)  

