#imports
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle
import re
from datetime import datetime
from vllm import LLM, SamplingParams
import pandas as pd
import ast
import json
import shutil


#function for reading the config file
def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


#function to extract scores present within the <score></score> tags
def extract_scores(texts):
    return_scores = []
    
    for text in texts:
        temp=[]
        for t in text:
            pattern = r'<score>(\d+)</score>'
            scores = re.findall(pattern, t)
            scores = [int(score) for score in scores]
            temp.append(scores)
        return_scores.append(temp)

    return return_scores


#function to extract text from vllm output format
def extract_text(outputs):
    return_text=[]
    for output in outputs:
        temp=[]
        for el in output.outputs:
            temp.append(el.text)
        return_text.append(temp)
    return return_text


#read the config file to extract user values
config_path = "config.json"
config = read_config(config_path)

GPU_DEVICE = config["gpu_device"]
MAX_TOKENS = config["max_tokens"]
SAMPLES = config["no_of_samples"]
DATA_PATH = config["data_path"]
PROMPT_DIR = config["prompt_dir"]
MODEL_NAME = config["model_name"]
TEMPERATURE = config["temperature"]
OUTPUT_DIR = config["output_path"]
# EVAL_CRITERIA = config["eval_criteria"]


#set the cuda device that should be visible
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#intialize the vllm by the model to be used
llm = LLM(model=MODEL_NAME)
sampling_params = SamplingParams(max_tokens=MAX_TOKENS, n=SAMPLES, temperature=TEMPERATURE)


#enlist the prompt files and remove the .ipynb_checkpoints
prompt_files = os.listdir(PROMPT_DIR)
if ".ipynb_checkpoints" in prompt_files:
    prompt_files.remove(".ipynb_checkpoints")
# if EVAL_CRITERIA!=None:
#     prompt_files=[EVAL_CRITERIA + ".txt"]

#intialize review and summary lists
reviews_list=[]
all_summary_files = []

#read the data file
df = pd.read_csv(DATA_PATH)

#get the list of reviews for all the products
for t in df.reviews.to_list():
    reviews_list.append(ast.literal_eval(t))

#get the list of summaries for all the products
for t in df.summaries.to_list():
    all_summary_files.append(ast.literal_eval(t))

#create a directory for the current run
current_datetime = datetime.now()
current_datetime_str = current_datetime.strftime("%Y%m%d%H%M%S")
folder_path = OUTPUT_DIR + current_datetime_str + "/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
print("Directory created: ", folder_path)

# copy config file to the directory
shutil.copy2(config_path, folder_path)

#main code
for prompt_file in prompt_files:
    original=[]
    metric = prompt_file.split(".txt")[0]
    #if metric in ["fl","sent","coh","fa","sp"]:
    #    continue
    
    print("Running for: ", prompt_file)
    
    #read the prompt 
    with open(PROMPT_DIR + prompt_file, 'r') as file:
        prompt = file.read()
        
#     #get the human annotated scores for different metrics
#     for t in df[metric].to_list():
#         original.extend(ast.literal_eval(t))
        
    #collect all the prompts for each product, summary and metric
    final_prompts=[]
    for reviews, summaries in zip(reviews_list, all_summary_files):
        for summary in summaries:
            text = prompt.format("\n".join(reviews),summary)
            final_prompts.append(text)

    #code to generate outputs using LLM
    outputs = llm.generate(final_prompts, sampling_params)

    #extract text and scores from the LLM outputs
    texts = extract_text(outputs)
    scores = extract_scores(texts)

    #intialize
    final_texts = []
    final_scores = []
    
    #ensure the scores collected are proper and remove irrelevant scores
    for te,sc in zip(texts, scores):
        temp_text = []
        temp_scores = []
        for t,s in zip(te,sc):
            if len(s)==1:
                temp_text.append(t)
                temp_scores.append(s[0])
        if len(temp_text)>=SAMPLES-5:
            final_texts.append(temp_text[:SAMPLES-5])
            final_scores.append(temp_scores[:SAMPLES-5])
        else:
            final_texts.append(temp_text)
            final_scores.append(temp_scores)
    
    
    output_text_path = folder_path + "output_text/"
    output_score_path = folder_path + "output_scores/"
#     original_score_path = folder_path + "original_scores/"
          
    if not os.path.exists(output_text_path):
        os.makedirs(output_text_path)
    if not os.path.exists(output_score_path):
        os.makedirs(output_score_path)
#     if not os.path.exists(original_score_path):
#         os.makedirs(original_score_path)

    with open(output_text_path + metric + ".pkl", 'wb') as file:
        pickle.dump(final_texts, file)

    with open(output_score_path + metric + ".pkl", 'wb') as file:
        pickle.dump(final_scores, file)

#     with open(original_score_path + metric + ".pkl", 'wb') as file:
#         pickle.dump(original, file)
