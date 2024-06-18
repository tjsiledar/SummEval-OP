import pickle
import os
from statistics import mean, median, stdev
from scipy.stats import spearmanr, kendalltau
import warnings
import json
import pandas as pd
import ast
import matplotlib.pyplot as plt
import shutil

mapping_dict = {
    "fl.pkl":"Fluency",
    "coh.pkl":"Coherence",
    "rel.pkl":"Relevance",
    "fa.pkl":"Faithfulness",
    "asp.pkl":"Aspect Coverage",
    "sent.pkl":"Sentiment Consistency",
    "sp.pkl":"Specificity"
}

def convert_ast(row):
    return ast.literal_eval(row)

def frequency_calculation(score_list):
    return_dict={1:0, 2:0, 3:0, 4:0, 5:0}
    for el in score_list:
        return_dict[el]+=1
    return return_dict
    
def probability_calculation(score_list):
    freq_dict = frequency_calculation(score_list)
    return freq_dict, {key: value / len(score_list) for key, value in freq_dict.items()}

def scoring_function(score_list, strategy):
#     print(score_list)
    if strategy=="mean":
        return mean(score_list)
    if strategy=="median":
        return median(score_list)
    if strategy=="geval":
        score=0
        freq_dict, prob_dict = probability_calculation(score_list)
        
        for el in freq_dict.keys():
            score += el*prob_dict[el]
        return score  
        

def create_plots(cmetrics, y_values, x_values, metrics, figname, nm):
    
    # Create figure and axes
    fig, axs = plt.subplots(nm, 2, figsize=(10, 15))
    
    for i, metric in enumerate(cmetrics):
        for j in range(nm):
            ax = axs[j, i] if nm > 1 else axs[i]
            
            # Plot data and correlation
            ax.plot(x_values[j], y_values[metric][j])
            ax.set_title(mapping_dict[metrics[j]] + " (" + metric + ")")
    # Adjust layout
    plt.tight_layout()
    plt.savefig("../plots/" + figname + ".png",  bbox_inches="tight",dpi=300)


# Suppress all warnings
warnings.simplefilter("ignore")

#function for reading the config file
def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

#read the config file to extract user values
config_path = "correlation.json"
config = read_config(config_path)

SAVE = config["save"]
STRATEGY = config["strategy"]
OUTPUT_PATH = config["dir_path"]
DATA_PATH = config["data_path"]
N = config["N"]
BATCH = config["batch"]
L = config["want_list"]
PLOT=config["plot"]
SAVE_PATH = config["save_path"]
TYPE = config["type"]

folder_path = os.listdir(OUTPUT_PATH)
if "config.json" in folder_path:
    folder_path.remove("config.json")

metrics = os.listdir(OUTPUT_PATH + "output_scores")

order_metrics = list(mapping_dict.keys())

no_summaries = len(ast.literal_eval(pd.read_csv(DATA_PATH).summaries[0]))

original_df = pd.read_csv(DATA_PATH)
original_df = original_df.applymap(convert_ast)


final_list={}
for metric in order_metrics:
    metric_name = metric.split(".pkl")[0]
    
    original_scores=[]
    for t in original_df[metric_name].to_list():
        original_scores.extend(t)
        
    if metric not in metrics:
        continue
    print()
    print("********* Running for ", metric, "***********")
    model_path = OUTPUT_PATH + "output_scores/" + metric
#     original_path = OUTPUT_PATH + "original_scores/" + metric
    
    with open(model_path, 'rb') as file:
        model_scores = pickle.load(file)
#     with open(original_path, 'rb') as file:
#         original_scores = pickle.load(file)
        
    hyp = [model_scores[i:i+no_summaries] for i in range(0, len(model_scores), no_summaries)]
    ref = [original_scores[i:i+no_summaries] for i in range(0, len(original_scores), no_summaries)]
    ref = [[mean(e) for e in el] for el in ref]
    
#     print(hyp)
#     print(ref)
    
    final_dict = {}
    
    for i in range(0, N, BATCH):
#         print(i,i+BATCH)
        if TYPE=="standard":
            if i==0:
                i=1
            final_dict[i] = {}
            hyp_ = [[scoring_function(e[:i], STRATEGY) for e in el] for el in hyp]
            sr_ = [spearmanr(ref[i], hyp_[i]).correlation for i in range(len(hyp_))]
            sr_ = [0 if str(s) == "nan" else s for s in sr_]
            kt_ = [kendalltau(ref[i], hyp_[i]).correlation for i in range(len(hyp_))]
            kt_ = [0 if str(s) == "nan" else s for s in kt_]
        if TYPE=="new":
            final_dict[i] = {}
            hyp_ = [[scoring_function(e[i:i+BATCH], STRATEGY) for e in el] for el in hyp]
            sr_ = [spearmanr(ref[i], hyp_[i]).correlation for i in range(len(hyp_))]
            sr_ = [0 if str(s) == "nan" else s for s in sr_]
            kt_ = [kendalltau(ref[i], hyp_[i]).correlation for i in range(len(hyp_))]
            kt_ = [0 if str(s) == "nan" else s for s in kt_]
        
        if L!="yes":
            final_dict[i]["spearman"] = mean(sr_)
            final_dict[i]["kendall"] = mean(kt_)
        else:
            final_dict[i]["spearman"] = sr_
            final_dict[i]["kendall"] = kt_
        
    final_list[metric] = final_dict
    
    print(json.dumps(final_dict, indent=4))
    
if SAVE=="yes":
    fold_no = OUTPUT_PATH.split("/")[-2]
    
    save_path = SAVE_PATH+fold_no + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # copy config file to the directory
    shutil.copy2(config_path, save_path)
    
    with open(save_path + "outputs.json", "w") as f:
        json.dump(final_list, f)

if PLOT=="yes":    
    plot_dict = {"spearman":[], "kendall":[]}
    x_values=[]

    for fl in final_list:
        plot_dict["spearman"].append([value["spearman"] for value in final_list[fl].values()])
        plot_dict["kendall"].append([value["kendall"] for value in final_list[fl].values()])
        x_values.append(list(final_list[fl].keys()))

    create_plots(["spearman","kendall"], plot_dict, x_values, metrics, OUTPUT_PATH.split("/")[-2], nm=len(final_list))
    
    
    
    

    
