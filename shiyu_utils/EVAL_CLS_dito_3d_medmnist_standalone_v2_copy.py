# %%
import monai 
from script_EVAL_fm import *
import json 


json_files = "config_EVAL_fm.json"
args = build_args_from_path(json_files)
encoder= define_instance(args,"encoder")
#encoder.load_state_dict(torch.load("model_encoder.pth"))
diff_unet = define_instance(args,"diff_unet")
#diff_unet.load_state_dict(torch.load("model_diff_unet.pth"))
noise_scheduler = define_instance(args,"noise_scheduler")
upscaler = define_instance(args,"upscaler")
device = define_instance(args,"device")
inferer = define_instance(args,"inferer")
dito = DiffusionModelWrapper(
    encoder=encoder,
    diff_unet=diff_unet,
    noise_scheduler=noise_scheduler,
    upscaler=upscaler,
    loss_fn=torch.nn.MSELoss(),
)
dito.to(device)

if False:
    encoder.load_state_dict(torch.load("model_encoder.pth"))
    diff_unet.load_state_dict(torch.load("model_diff_unet.pth"))
else:
    dito.load_state_dict(torch.load("dito.pth"))

# %%
data_root = "/data4/brats25/brats2025_07_30_buildfromzero_for_project_brats25/ZHANGYUXUAN_SYN_ADNI/ADNI_1"
import glob
import os 
import torchio as tio 
data_list =sorted(glob.glob(os.path.join(data_root,"*.nii*")))
data_list = [{"image":d} for d in data_list]
total_length = len(data_list)
train_list, val_list,test_list = data_list[:int(total_length*0.8)],data_list[int(total_length*0.8):int(total_length*0.9)],data_list[int(total_length*0.9):]
print(len(train_list),len(val_list),len(test_list))
from maisi_vae_transform import define_mri_vae_transform


# %%
transform = define_mri_vae_transform(
    is_train=False,
    random_aug=True,
    val_patch_size=(256,256,256),
    spacing_type="fixed",
    spacing=(1.0,1.0,1.0),
    image_keys=["image"],
)
train_dataset = monai.data.Dataset(train_list,transform)
dataloader = monai.data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)
batch = next(iter(dataloader))
image = batch["image"].to(device)
tio.ScalarImage(tensor=image[0]).plot()

# %%
#args.inferer["roi_size"]=[16,16,16]
inferer = define_instance(args,"inferer")
latent_dir = "/data4/brats25/dito_2d3d_demo/demos/demos_data/data_for_exp_1_basic_copy_infer"
if len(glob.glob(latent_dir+"/*.pt"))>=len(train_list):
    print("latent_dir already exists")
else:
    os.makedirs(latent_dir,exist_ok=True)
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader):
            image = batch["image"].to(device)
            latent = inferer (inputs=image,network=dito.encode)
            basename = os.path.basename(image.meta["filename_or_obj"][0]).replace(".gz","").replace(".nii",".pt")
        basename = "latent_"+basename
        filename = os.path.join(latent_dir,basename)
        torch.save(latent,filename)
        
# args.inferer["roi_size"]=[16,16,16]
# inferer_recon = define_instance(args,"inferer")
# with torch.no_grad(), torch.cuda.amp.autocast():
#     latent = glob.glob(latent_dir+"/*.pt")[0]
#     latent = torch.load(latent).to(next(iter(dito.parameters())).device)
#     recon  = inferer_recon(
#         inputs=latent.float(),
#         network=dito.decode,
#     )    

# %%
latent = sorted(glob.glob(latent_dir+"/*.pt")[0])

# %%
import pandas as pd
from pprint import pprint
from IPython.display import display

# Define the categorization function with improved logic
def categorize_terms(terms_list):
    """
    Categorizes a list of medical/research terms based on predefined keywords.

    Args:
        terms_list (list): A list of strings, where each string is a term to categorize.

    Returns:
        dict: A dictionary where keys are category names and values are lists of terms
              belonging to that category.
    """
    # Clean and get unique terms to avoid duplicate processing and ensure neat output
    cleaned_terms = [term.strip() for term in terms_list if term.strip()]
    unique_terms = sorted(list(set(cleaned_terms))) # Sort for consistent output order

    # Initialize categories dictionary
    categories = {
        "Study and Participant Identifiers": [],
        "Demographics and Baseline Characteristics": [],
        "Longitudinal Data/Time Points": [],
        "Biomarkers": [], # Renamed for clarity (includes CSF and PET)
        "Everyday Cognition (ECog) Questionnaires": [],
        "Cognitive Assessments (Scores/Scales)": [],
        "Neuroimaging": [], # Renamed for clarity
        "Uncategorized": [] # To catch any terms that don't fit into defined categories
    }

    # Define keywords for each category.
    # IMPORTANT: The order of 'if/elif' checks in the loop below is crucial for correct classification.
    # More specific categories should be checked first.
    category_keywords = {
        "Everyday Cognition (ECog) Questionnaires": ["ecogpt", "ecogsp", "ecog"],
        "Biomarkers": ["abeta", "tau", "ptau", "fdg", "pib", "av45", "fbb"],
        "Neuroimaging": [
            "ucsf", "ventricles", "hippocampus", "wholebrain", "entorhinal", "fusiform",
            "midtemp", "icv", 
            "fldstreng", 
            #"fsversion", #"imageuid",
            "mri", "pet", "suvr" # General terms, but specific enough in context
        ],
        "Cognitive Assessments (Scores/Scales)": [
            "cdr", "adas", "mmse", "ravlt", "faq", "moca", "pacc",
            "digitscor", "trabscor", "ldeltotal", "adasq4",
            "recall", "learning", "forgetting", "composite"
        ],
        "Longitudinal Data/Time Points": ["month", "m"], # Only explicit time components for longitudinal
        "Study and Participant Identifiers": ["rid", "colprot", "origprot", "ptid", "site", "viscode", "update_stamp"],
        "Demographics and Baseline Characteristics": [
            "dx", "examdate", "age", "ptgender", "pteducat", "ptethcat", "ptraccat", "ptmarry", "apoe4"
        ]
    }

    # Process each unique term
    for term in unique_terms:
        lower_term = term.lower()
        assigned = False

        # Apply checks in order of specificity/priority
        if any(keyword in lower_term for keyword in category_keywords["Everyday Cognition (ECog) Questionnaires"]):
            categories["Everyday Cognition (ECog) Questionnaires"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Biomarkers"]):
            categories["Biomarkers"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Neuroimaging"]):
            categories["Neuroimaging"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Cognitive Assessments (Scores/Scales)"]):
            categories["Cognitive Assessments (Scores/Scales)"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Longitudinal Data/Time Points"]):
            categories["Longitudinal Data/Time Points"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Study and Participant Identifiers"]):
            categories["Study and Participant Identifiers"].append(term)
            assigned = True
        elif any(keyword in lower_term for keyword in category_keywords["Demographics and Baseline Characteristics"]):
            categories["Demographics and Baseline Characteristics"].append(term)
            assigned = True
        else:
            categories["Uncategorized"].append(term)

    # Sort terms within each category alphabetically for clean output
    for category in categories:
        categories[category] = sorted(list(set(categories[category])))

    return categories

# --- Your DataFrame processing begins here ---
df = pd.read_csv("ADNIMERGE_03Aug2025.csv")

# Filter out columns ending with "_bl" as per your instruction
df = df.drop(columns=[c for c in df.columns if c.endswith("_bl")])

# Select only first visit data, sort by EXAMDATE for each PTID
df = df.sort_values(by=["PTID","EXAMDATE"])
df = df.drop_duplicates(subset=["PTID"],keep="first")

# Get the current columns of the filtered DataFrame
df_columns_array = df.columns.values

# Convert the numpy array to a list for categorization
terms_list_for_categorization = df_columns_array.tolist()

# Categorize the terms
categorized_df_columns = categorize_terms(terms_list_for_categorization)

print("--- Categorization of Current DataFrame Columns (after filtering '_bl' and first visit) ---")
display(categorized_df_columns) # Print the raw categorization dictionary for verification


# --- Now, integrate categories into the completeness DataFrame ---

# Compute completeness for each column
completeness = (df.notnull().sum() / len(df)) * 100
completeness_df = pd.DataFrame({'Column': completeness.index, 'Completeness (%)': completeness.values})
completeness_df = completeness_df.sort_values(by='Completeness (%)', ascending=False).reset_index(drop=True)


# Create a mapping dictionary from column name to its category
column_to_category_map = {}
for category, terms_in_category in categorized_df_columns.items():
    for term in terms_in_category:
        column_to_category_map[term] = category

# Add a 'Category' column to the completeness_df using the mapping
completeness_df['Category'] = completeness_df['Column'].map(column_to_category_map)

# Handle potential NaNs in 'Category' if some columns weren't categorized
completeness_df['Category'].fillna('Uncategorized', inplace=True)


print("\n--- Detailed Completeness with Categorization ---")
# Display the combined DataFrame
display(completeness_df)

df = pd.read_csv("ADNIMERGE_03Aug2025.csv")

# Filter out columns ending with "_bl" as per your instruction
df = df.drop(columns=[c for c in df.columns if c.endswith("_bl")])

# Select only first visit data, sort by EXAMDATE for each PTID
df = df.sort_values(by=["PTID","EXAMDATE"])
df = df.drop_duplicates(subset=["PTID"],keep="first")

id_column_types = [
    'Study and Participant Identifiers',
    "Demographics and Baseline Characteristics",   
    ]
id_columns = [column for id_column_type in id_column_types for column in categorized_df_columns[id_column_type]]
scan_column_types = [
    'Neuroimaging',
]
scan_columns = [column for scan_column_type in scan_column_types for column in categorized_df_columns[scan_column_type]]
GT_column_types = [
    'Biomarkers',
]
GT_columns = [column for GT_column_type in GT_column_types for column in categorized_df_columns[GT_column_type]]
cognition_column_types=[
    "Everyday Cognition (ECog) Questionnaires",
    'Cognitive Assessments (Scores/Scales)',
]
cognition_columns = [column for cognition_column_type in cognition_column_types for column in categorized_df_columns[cognition_column_type]]
drop_column_types = [
    'Longitudinal Data/Time Points',
    'Uncategorized'
]
drop_columns = [column for drop_column_type in drop_column_types for column in categorized_df_columns[drop_column_type]]
columns_total = [
    id_columns,
    scan_columns,
    GT_columns,
    cognition_columns,
    drop_columns,
]

for columns in columns_total:
    display(completeness_df[completeness_df["Column"].isin(columns)].reset_index(drop=True))


# %%
latents = sorted(glob.glob(latent_dir+"/*.pt"))
latent_ids = [os.path.basename(latent).split(".")[0][7:] for latent in latents]
print(latent_ids[:5],latent_ids[-5:],len(latent_ids))


#display(df[["AGE","PTID","DX"]])
df = pd.read_csv("ADNIMERGE_03Aug2025.csv")
org_df = df.copy()
df = df.drop(columns=[c for c in df.columns if c.endswith("_bl")])
df = df.sort_values(by=["PTID","EXAMDATE"])
df = df.drop_duplicates(subset=["PTID"],keep="first")
df = df.dropna(subset=["DX"])
df_latent = df[df["PTID"].isin(latent_ids)]
other_cols = ["AGE"]
display(df_latent[["PTID","DX"]+other_cols])

subject_data_map = df_latent.set_index("PTID")[["DX", "AGE"]].T.to_dict('list')
label_map = {"CN":0,"MCI":1,"Dementia":2}
# 2. Construct datalist using the efficient map
datalist = []
for latent, latent_id in zip(latents, latent_ids):
    # Check if the latent file exists and if its ID is present in our subject_data_map
    if os.path.exists(latent) and latent_id in subject_data_map:
        # Perform efficient lookup to get DX and AGE
        dx, age = subject_data_map[latent_id]
        datalist.append({
            "subject_id": latent_id,
            "image": latent,
            "labels": label_map[dx],
            "age": age,
        })

print(f"\nLengths: datalist={len(datalist)}, df_latent={len(df_latent)}, latents={len(latents)}")

print("\nFirst 3 entries of the datalist (for verification):")
pprint(datalist[:3]) # Using pprint for cleaner dictionary output



# %%
# by hand check if EXAMDATA is earliest for each PTID
display(org_df[org_df["PTID"]=="011_S_0003"])
display(df[df["PTID"]=="011_S_0003"])
pd.DataFrame(datalist)["labels"].value_counts()

# %%
transform = lambda x: torch.load(x,map_location="cpu")[0].float()
#transform(datalist[0]).shape
import monai.data as md
import monai.transforms as mt
dataset = md.Dataset(datalist,transform=mt.Lambdad(keys=["image"],func=transform))
dataloader = md.ThreadDataLoader(dataset,batch_size=1,shuffle=False)

for batch in tqdm(dataloader):
    pass

# %%
from script_EVAL_fm import VaeEncoder,DiffusionDecoder,ConditionalFlowMatching,LatentConditionalFlowMatching
from monai.inferers import DiffusionInferer

eval_cond_diff_arg_path = "config_EVAL_fm_cond_diff_args.json"
TRAIN_START = not os.path.exists(eval_cond_diff_arg_path)
model_path = eval_cond_diff_arg_path.replace(".json","_pipeline.pt")
TRAIN_FINISHED = os.path.exists(model_path)

if TRAIN_START:
    print(f"json {os.path.abspath(eval_cond_diff_arg_path)} not found, start new train")
    args = build_args_from_path("/data4/brats25/dito_2d3d_demo/demos/exp_1_basic/exp_1.json")
    cond_diff_unet = args.diff_unet.copy()
    cond_diff_unet["in_channels"]=4
    cond_diff_unet["out_channels"]=4
    cond_diff_unet["cross_attention_dim"]=16
    cond_diff_unet["with_conditioning"]=True
    setattr(args,"cond_diff_unet",cond_diff_unet)

    import json 
    with open(eval_cond_diff_arg_path,"w") as f:
        json.dump(args.__dict__,f,indent=4)
    pprint(f"save new json to {os.path.abspath(eval_cond_diff_arg_path)}")
    pprint(args.__dict__)
else:
    args = build_args_from_path(eval_cond_diff_arg_path)
    print(f"load json from {os.path.abspath(eval_cond_diff_arg_path)}")
    print("*"*100)
    pprint(args.__dict__)
    print("*"*100)
    TRAIN_FINISHED=False

vae_encoder = VaeEncoder(encoder)
diffusion_decoder = DiffusionDecoder(diff_unet,noise_scheduler,upscaler)

diff_inferer = DiffusionInferer(scheduler=noise_scheduler)

cond_diff_unet = define_instance(args,"cond_diff_unet")

# pipeline = ConditionalFlowMatching(
#     encoder= vae_encoder,
#     diffusion_model=cond_diff_unet,
#     inferer=diff_inferer,
#     decoder= diffusion_decoder,
#     num_class=3,
# )
pipeline = LatentConditionalFlowMatching(
    encoder= vae_encoder,
    diffusion_model=cond_diff_unet,
    inferer=diff_inferer,
    decoder= diffusion_decoder,
    num_class=3,
)

if TRAIN_FINISHED:
    print(f"loading existing model from {os.path.abspath(model_path)}")
    print("train will not start, will do inference only")
    pipeline.load_state_dict(torch.load(model_path))
else:
    print(f"WIll start train model in {os.path.abspath(model_path)}")


try:
    pipeline.label_embed(torch.randint(0,5,(100,))) # 3 class + 1 uncond  = 0,1,2,3  randint : 0,1,2,3,4  4 is not allowed 
except:
    print("Red Tests pass!")
    print("# 3 class + 1 uncond  = 0,1,2,3  randint : 0,1,2,3,4  4 is not allowed ")

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
if not TRAIN_FINISHED :
    print("starting new train ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    pipeline = LatentConditionalFlowMatching(vae_encoder,diff_inferer,cond_diff_unet,diffusion_decoder)
    pipeline.to("cuda")
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-4)

    # To store loss values for plotting
    loss_history = [] 
else:
    print("train will not start ")



# %%
if not TRAIN_FINISHED:
    try:
        print("Starting training...")
        # The full training loop
        for epoch in range(100):
            # Set models to training mode at the start of each epoch
            pipeline.train()
            
            for i, batch in enumerate(dataloader):
                # Move data to the correct device
                image = batch["image"].to(device)
                label = batch["labels"].to(device).long()
                label += 1 
                
                optimizer.zero_grad()
                
                # Use autocast for mixed-precision training
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    loss = pipeline(image,label)

                # Scale the loss and backpropagate
                scaler.scale(loss).backward()
                
                # Update the weights
                scaler.step(optimizer)
                
                # Update the scaler for the next iteration
                scaler.update()
                
                # Record loss
                current_loss = loss.item()
                loss_history.append(current_loss)

                # Console logging
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/100], Step [{i}/{len(dataloader)}], Loss: {current_loss:.4f}")
                    if epoch % 2 ==0 :
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            pipeline.eval()
                            generated_image = pipeline.sample_latents(label[0:1],cfg=3)
                            subject = tio.Subject(
                                gen = tio.ScalarImage(tensor=generated_image[0].detach().cpu()),
                                real = tio.ScalarImage(tensor=image[0].detach().cpu()),
                            )
                            subject.plot(title=f"epoch {epoch}",vmin=0,vmax=1,figsize=(10,10),cmap_dict={"org":"gray","gen":"gray","diff":"jet"})
                        # plot loss history 
                        plt.figure(figsize=(10, 5))
                        plt.plot(loss_history, label='Training Loss')
                        plt.xlabel('Step')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.show()
                        torch.save(pipeline.state_dict(),"pipeline.pt")


        print("Training finished.")
        torch.save(pipeline.state_dict(),model_path)
    except Exception as e:
        print("Training interrupted.")
        torch.save(pipeline.state_dict(),"pipeline_interrupted.pt")
        import os 
        os.system("nvidia-smi")
else:
    pipeline.load_state_dict(torch.load(model_path))


