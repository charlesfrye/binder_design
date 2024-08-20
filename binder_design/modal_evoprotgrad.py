import os
from modal import App, Secret, gpu, Image, enter, method
import logging
from datetime import datetime
from binder_design import DATA_DIR, EGFS, EGFR, EVO_PROT_GRAD_RESULTS_DIR, FOLD_RESULTS_DIR
from binder_design.utils import get_mutation_diff, hash_seq
import pandas as pd
from datetime import datetime

import time

def get_fold_results():
    fold_csvs = list(FOLD_RESULTS_DIR.glob('*.csv'))
    fold_df = pd.concat([pd.read_csv(csv) for csv in fold_csvs]).reset_index(drop=True)
    return fold_df

def get_folded_ids():
    fold_df = get_fold_results()
    return fold_df['seq_id'].unique().tolist()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
image = (
    Image
    .debian_slim(python_version="3.10")
    .pip_install('uv')
    .run_commands("uv pip install --system --compile-bytecode torch evo_prot_grad numpy", gpu="a10g")
    .run_commands("uv pip install --system --compile-bytecode pandas scikit-learn", gpu="a10g")
    )

app = App(name="evo_prot_grad", image=image)

with image.imports():
    import torch
    import numpy as np
    import pandas as pd
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import KFold
    
  
@app.function(
    container_idle_timeout=150,
    image=image,
    gpu="a10g",
    # concurrency_limit=20,
    timeout=9600,
)
def train_and_sample_evo_prot_grad(
    input_seqs: list,
    fold_results: list,
    n_new_seqs_to_return:int=100,
    n_serial_chains_per_seq:int=10,
    n_steps:int=20,
    max_mutations:int=4,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import KFold
    from evo_prot_grad.models.downstream_cnn import OneHotCNN
    from evo_prot_grad.common.tokenizers import OneHotTokenizer
    from evo_prot_grad.common.utils import CANONICAL_ALPHABET
    from evo_prot_grad import get_expert
    import evo_prot_grad
    
    logger.info("Starting train_and_sample_evo_prot_grad function")
    
    class ProteinDataset(Dataset):
        def __init__(self, sequences, properties):
            self.sequences = sequences
            self.properties = properties

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.properties[idx]
        
    fold_results = pd.DataFrame(fold_results)
    df = fold_results.query('pae_interaction < 10').drop_duplicates(subset=['seq_id']).sort_values('pae_interaction')

    df['len'] = df['binder_sequence'].apply(len)
    df = df.query('len == 50')
    all_sequences = df['binder_sequence'].to_list()
    all_properties = (-df['pae_interaction']).to_list() # negative bc too lazy to edit evo_prot_grad
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
   
    # Set up k-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameters
    batch_size = 32
    num_epochs = 500
    learning_rate = 3e-4

    logger.info(f"Hyperparameters - batch_size: {batch_size}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")

    # Lists to store results
    all_train_losses = []
    all_val_losses = []
    all_models = []  # List to store models

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_sequences)):
        logger.info(f"Starting fold {fold + 1}/{n_splits}")
        
        # Split data into train and validation sets
        train_seq = [all_sequences[i] for i in train_idx]
        train_prop = [all_properties[i] for i in train_idx]
        val_seq = [all_sequences[i] for i in val_idx]
        val_prop = [all_properties[i] for i in val_idx]
        
        # Create datasets and dataloaders
        train_dataset = ProteinDataset(train_seq, train_prop)
        val_dataset = ProteinDataset(val_seq, val_prop)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model, loss function, and optimizer
        model = OneHotCNN(vocab_size=20, kernel_size=8, input_size=64).to(device)
        tokenizer = OneHotTokenizer(alphabet=CANONICAL_ALPHABET)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for sequences, properties in train_loader:
                inputs = tokenizer(sequences).to(device)
                properties = properties.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), properties.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for sequences, properties in val_loader:
                    inputs = tokenizer(sequences).to(device)
                    properties = properties.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), properties.float())
                    epoch_val_loss += loss.item()
            
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(epoch_val_loss / len(val_loader))
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_models.append(model)
        
        # Training set predictions
        model.eval()
        train_predictions = []
        train_actual_values = []
        with torch.no_grad():
            for sequences, properties in train_loader:
                inputs = tokenizer(sequences).to(device)
                outputs = model(inputs)
                if outputs.dim() == 0:
                    train_predictions.append(outputs.item())
                elif outputs.dim() == 1 and len(outputs) == 1:
                    train_predictions.append(outputs[0].item())
                else:
                    train_predictions.extend(outputs.squeeze().cpu().tolist())
                train_actual_values.extend(properties.tolist())
        
        
        # Validation set predictions
        val_predictions = []
        val_actual_values = []
        with torch.no_grad():
            for sequences, properties in val_loader:
                inputs = tokenizer(sequences).to(device)
                outputs = model(inputs)
                if outputs.dim() == 0:
                    val_predictions.append(outputs.item())
                elif outputs.dim() == 1 and len(outputs) == 1:
                    val_predictions.append(outputs[0].item())
                else:
                    val_predictions.extend(outputs.squeeze().cpu().tolist())
                val_actual_values.extend(properties.tolist())
        
        
        mse = mean_squared_error(val_actual_values, val_predictions)
        mae = mean_absolute_error(val_actual_values, val_predictions)
        r_squared = r2_score(val_actual_values, val_predictions)
        
        logger.info(f"Validation set - MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r_squared:.4f}")
        
    logger.info("Finished training, starting sampling")
    
    experts = []
    for model in all_models:
        regression_expert = get_expert(
            'onehot_downstream_regression',
            temperature = 1.0,
            scoring_strategy = 'attribute_value',
            model = model
            )
        experts.append(regression_expert)

    all_preds = []
    for seq in input_seqs:
        try:
            input_pae_i = (df.query('binder_sequence == @seq')['pae_interaction']).values[0]
        except:
            logger.warning(f"Sequence {seq} not found in fold results, skipping")
            continue
        for ii in range(n_serial_chains_per_seq):
            # logger.info(f"Starting chain {ii+1}/{n_serial_chains_per_seq} for sequence {seq}")
            variants, scores = evo_prot_grad.DirectedEvolution(
                            wt_protein = seq,    # path to wild type fasta file
                            output = 'all',                # return best, last, all variants    
                            experts = experts,   # list of experts to compose
                            parallel_chains = 1,            # number of parallel chains to run
                            n_steps = n_steps,                   # number of MCMC steps per chain
                            max_mutations = max_mutations,             # maximum number of mutations per variant
                            verbose = False                # print debug info to command line
            )()
            prop_seqs = [''.join(v[0].split(' ')) for v in variants]
            pred_pae_i = -(scores - input_pae_i).ravel()
            preds = pd.DataFrame({
                'binder_sequence':prop_seqs, 
                'pred_pae_interaction': pred_pae_i,
                'parent_sequence': seq,
                'seq_id': seq,
                })
            preds['chain'] = ii
            
            all_preds.append(preds)
            
    all_preds = pd.concat(all_preds).drop_duplicates(subset=['seq_id']).reset_index(drop=True)
    all_preds = all_preds.sort_values('pred_pae_interaction')
    logger.info(f"Finished sampling, returning top {n_new_seqs_to_return} sequences")
    n_new_seqs_to_return = min(n_new_seqs_to_return, len(all_preds))
    return all_preds.head(n_new_seqs_to_return).to_dict(orient='records')
    # return all_preds.head(n_new_seqs_to_return)['binder_sequence'].to_list()

@app.local_entrypoint()
def test():
    logger.info("Starting test function")
    fold_results = get_fold_results().drop(columns=['pdb_content'])
    df = fold_results.query('pae_interaction < 10').drop_duplicates(subset=['seq_id']).sort_values('pae_interaction').head(10)
    logger.info(f"Test dataframe:\n{df}")
    input_seqs = df['binder_sequence'].to_list()
    
    logger.info("Calling train_and_sample_evo_prot_grad")
    output = train_and_sample_evo_prot_grad.remote(
        input_seqs=input_seqs, 
        fold_results=fold_results.to_dict(orient='records')
        )
    output = pd.DataFrame(output)
    logger.info(f"Output:\n{output}")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = EVO_PROT_GRAD_RESULTS_DIR / f"evo_prot_grad_results_{now}.csv"
    output.to_csv(fp, index=False)
    logger.info(f"Output saved to {fp}")
    