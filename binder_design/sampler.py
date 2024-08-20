
from typing import List, Tuple
import numpy as np
from modal import App, Secret, gpu, Image, enter, method
import modal
import logging
from datetime import datetime
from binder_design import (
    DATA_DIR, EGFS, EGFR, FOLD_RESULTS_DIR, PEPMLM_RESULTS_DIR, COLABFOLD_GPU_CONCURRENCY_LIMIT, TEMPLATE_A3M_PATH, EVO_PROT_GRAD_RESULTS_DIR)
from binder_design.utils import get_mutation_diff, hash_seq, get_mlm_results
import pandas as pd
import re

from modal_colabfold import app as colabfold_app
from modal_pepmlm import app as pepmlm_app
from modal_evoprotgrad import app as evoprotgrad_app

def get_fold_results():
    fold_csvs = list(FOLD_RESULTS_DIR.glob('*.csv'))
    fold_df = pd.concat([pd.read_csv(csv) for csv in fold_csvs]).reset_index(drop=True)
    return fold_df

def get_folded_ids():
    fold_df = get_fold_results()
    return fold_df['seq_id'].unique().tolist()

app = App(name='binder_sampling')
app.include(colabfold_app)
app.include(pepmlm_app)
app.include(evoprotgrad_app)


parallel_edit_binders = modal.Function.lookup("pepmlm", 'parallel_edit_binders')
parallel_fold_and_extract = modal.Function.lookup("colabfold", 'parallel_fold_and_extract')
train_and_sample_evo_prot_grad = modal.Function.lookup("evo_prot_grad", 'train_and_sample_evo_prot_grad')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.local_entrypoint()
def evolve_binders(
    init_binder_seqs=None,
    init_from_folded:bool=True,
    init_from_folded_top_k:int=5,
    target_seq=EGFR, 
    frac_residues_to_mask=0.05, 
    top_k=8, 
    num_variations_per_binder=50, 
    min_n_binder_seqs=10,
    n_generations=4,
    n_pepmlm_survivors=50,
    pepmlm_top_k:int = 100,
    n_colabfold_survivors=20,
    select_from_all_folded:bool=True,
    folded_top_k:int = 20,
):
    from binder_design.utils import get_mutation_diff
    logger.info("Starting evolve_binders function")
    if init_from_folded:
        init_df = get_fold_results().drop_duplicates(subset=['seq_id']).sort_values(by='pae_interaction', ascending=True)
        if init_from_folded_top_k is not None:
            init_df = init_df.head(init_from_folded_top_k)
            # sample based on pae_interaction
            p = init_df['pae_interaction'].values / init_df['pae_interaction'].sum()
            init_df = init_df.sample(n=min_n_binder_seqs, replace=True, weights=p)
        init_binder_seqs = init_df['binder_sequence'].tolist()

        
    if init_binder_seqs is None:
        init_binder_seqs = [EGFS]
    
    # Ensure we have at least min_n_binder_seqs to start with
    if len(init_binder_seqs) < min_n_binder_seqs:
        multiplier = -(-min_n_binder_seqs // len(init_binder_seqs))  # Ceiling division
        init_binder_seqs = init_binder_seqs * multiplier
        logger.info(f"Expanded initial binder sequences from {len(init_binder_seqs) // multiplier} to {len(init_binder_seqs)} to meet minimum requirement.")

    # Log initial sequences
    logger.info(f"Initial binder sequences:")
    for i, seq in enumerate(init_binder_seqs):
        logger.info(f"  Sequence {i+1}: {seq}")
    logger.info(f"Total number of initial sequences: {len(init_binder_seqs)}")
    current_generation = init_binder_seqs
    
    for generation in range(n_generations):
        logger.info(f"Starting generation {generation + 1}/{n_generations}")
        
        # Generate variations for the current generation
        logger.info("Generating variations for the current generation")
        results = parallel_edit_binders.remote(current_generation, target_seq, frac_residues_to_mask, top_k, num_variations_per_binder)
        
        # Sort the results by perplexity (lower is better)
        sorted_results = results.sort_values(by='unmasked_ppl')
        
        # dedupe
        sorted_results = sorted_results.drop_duplicates(subset=['seq_id'])
        
        # Save this generation's results
        save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = PEPMLM_RESULTS_DIR / f"pepmlm_binders_gen{generation+1}_{save_time}.csv"
        sorted_results.to_csv(filepath, index=False)
        logger.info(f"Saved PepMLM generation {generation + 1} results to {filepath}")
        
        # Print some stats about this generation
        logger.info(f"Generation {generation + 1} stats:")
        logger.info(f"  Best PPL: {sorted_results['unmasked_ppl'].min()}")
        logger.info(f"  Median PPL: {sorted_results['unmasked_ppl'].median()}")
        logger.info(f"  Worst PPL: {sorted_results['unmasked_ppl'].max()}")
        
        # Select the top n_survivors as the next generation
        already_folded = get_folded_ids()
        logger.info(f"Total N folded = {len(already_folded)} sequences")
        sorted_results = sorted_results[~sorted_results['seq_id'].isin(already_folded)]
        if pepmlm_top_k is not None:
            pepmlm_top_k = min(pepmlm_top_k, n_pepmlm_survivors)
            sorted_results = sorted_results.head(pepmlm_top_k).sample(n=n_pepmlm_survivors)
        seqs_to_fold = sorted_results.set_index('seq_id')['binder'].head(n_pepmlm_survivors).to_dict()
        # Sanitize sequence IDs to remove slashes
        sanitized_seqs_to_fold = {}
        for seq_id, seq in seqs_to_fold.items():
            sanitized_id = str(seq_id).replace('/', '_')
            sanitized_seqs_to_fold[sanitized_id] = seq
        seqs_to_fold = sanitized_seqs_to_fold

        logger.info("Sanitized sequence IDs to remove slashes")
        for name, seq in seqs_to_fold.items():
            logger.info(f"{name}: {seq}")
        logger.info(f"Folding {len(seqs_to_fold)} sequences")
        batch_size = int(n_pepmlm_survivors / COLABFOLD_GPU_CONCURRENCY_LIMIT)
        batch_size = max(batch_size, 1)
        fold_results = parallel_fold_and_extract.remote(
            binder_sequences=seqs_to_fold,
            template_a3m_path=None,
            target_seq=None,
            batch_size=batch_size,
            num_models=1, 
            num_recycle=1)
        
        fold_df = pd.DataFrame(fold_results)
        # fold_df['mut_str'] = fold_df['binder_sequence'].apply(get_mutation_diff, seq2=EGFS)
        fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
        fold_df = fold_df.sort_values(by='pae_interaction', ascending=True).reset_index(drop=True)
        fold_df['generation'] = generation + 1
        
        # Save the fold results
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
        fold_df.to_csv(fp, index=False)
        logger.info(f"Saved fold results to {fp}")
        
        if select_from_all_folded:
            fold_df = pd.concat([fold_df, get_fold_results()])
        fold_df = fold_df.drop_duplicates(subset=['seq_id']).sort_values(by='pae_interaction', ascending=True)
        
        # Log statistics for pae_interaction
        logger.info(f"  Best (lowest) PAE Interaction: {fold_df['pae_interaction'].min():.4f}")
        
        # Get the 2nd, 5th, and 10th best PAE Interactions
        second_best_pae = fold_df['pae_interaction'].nsmallest(2).iloc[-1]
        fifth_best_pae = fold_df['pae_interaction'].nsmallest(5).iloc[-1]
        tenth_best_pae = fold_df['pae_interaction'].nsmallest(10).iloc[-1]
        logger.info(f"  2nd Best PAE Interaction: {second_best_pae:.4f}")
        logger.info(f"  5th Best PAE Interaction: {fifth_best_pae:.4f}")
        logger.info(f"  10th Best PAE Interaction: {tenth_best_pae:.4f}")
        
        if folded_top_k is not None:
            fold_df = fold_df.head(folded_top_k)
            # sample based on pae_interaction
            p = fold_df['pae_interaction'].values / fold_df['pae_interaction'].sum()
            fold_df = fold_df.sample(n=n_colabfold_survivors, replace=False, weights=p)
            
        next_generation = fold_df.head(n_colabfold_survivors)['binder_sequence'].tolist()
        
        # Update current_generation for the next iteration
        current_generation = next_generation
        logger.info(f"Selected {len(current_generation)} sequences for the next generation")
        
        # Log sequences for next generation
        logger.info("Sequences selected for next generation:")
        for seq in current_generation:
            logger.info(f"{seq}")
        
        
        mut_strs = []
        for seq in current_generation:
            try:
                mut_str = get_mutation_diff(seq, EGFS)
                mut_strs.append(mut_str)
            except Exception as e:
                logger.warning(f"Failed to calculate mut_str for sequence: {seq}. Error: {str(e)}")
                mut_strs.append(None)
        try:
            for mut_str in mut_strs:
                logger.info(f"{mut_str}")
        except Exception as e:
            logger.error(f"Failed to log mut_strs. Error: {str(e)}")
            
    
    logger.info("Evolution complete.")
    return sorted_results


@app.local_entrypoint()
def evoprotgrad_binders(
    init_binder_seqs=None,
    init_from_folded:bool=True,
    init_from_folded_top_k:int=8,
    target_seq=EGFR, 
    min_n_binder_seqs=40,
    n_generations=8,
    n_evoprotgrad_survivors=320,
    n_colabfold_survivors=160,
    select_from_all_folded:bool=True,
    folded_top_k:int = 160,
    n_new_seqs_to_return:int=320,
    n_serial_chains_per_seq:int=5,
    n_steps:int=30,
    max_mutations:int=7,
):
    from binder_design.utils import get_mutation_diff
    import pandas as pd
    import re
    logger.info("Starting evoprotgrad_binders function")
    if init_from_folded:
        init_df = get_fold_results().drop_duplicates(subset=['seq_id']).sort_values(by='pae_interaction', ascending=True)
        if init_from_folded_top_k is not None:
            init_df = init_df.head(init_from_folded_top_k)
            # sample based on pae_interaction
            p = init_df['pae_interaction'].values / init_df['pae_interaction'].sum()
            init_df = init_df.sample(n=min_n_binder_seqs, replace=True, weights=p)
        init_binder_seqs = init_df['binder_sequence'].tolist()

    if init_binder_seqs is None:
        init_binder_seqs = [EGFS]
    
    # Ensure we have at least min_n_binder_seqs to start with
    if len(init_binder_seqs) < min_n_binder_seqs:
        multiplier = -(-min_n_binder_seqs // len(init_binder_seqs))  # Ceiling division
        init_binder_seqs = init_binder_seqs * multiplier
        logger.info(f"Expanded initial binder sequences from {len(init_binder_seqs) // multiplier} to {len(init_binder_seqs)} to meet minimum requirement.")

    # Log initial sequences
    logger.info(f"Initial binder sequences:")
    for i, seq in enumerate(init_binder_seqs):
        logger.info(f"  Sequence {i+1}: {seq}")
    logger.info(f"Total number of initial sequences: {len(init_binder_seqs)}")
    current_generation = init_binder_seqs
    
    for generation in range(n_generations):
        logger.info(f"Starting generation {generation + 1}/{n_generations}")
        
        # Get all fold results
        all_fold_results = get_fold_results().to_dict(orient='records')
        
        # Generate variations using EvoProt Grad
        logger.info("Generating variations using EvoProt Grad")
        evoprotgrad_results = train_and_sample_evo_prot_grad.remote(
            input_seqs=current_generation,
            fold_results=all_fold_results,
            n_new_seqs_to_return=n_new_seqs_to_return,
            n_serial_chains_per_seq=n_serial_chains_per_seq,
            n_steps=n_steps,
            max_mutations=max_mutations,
        )
        
        evoprotgrad_df = pd.DataFrame(evoprotgrad_results)
        logger.info(f"returned {len(evoprotgrad_df)} sequences")
        
        # Save this generation's results
        save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = EVO_PROT_GRAD_RESULTS_DIR / f"evoprotgrad_binders_gen{generation+1}_{save_time}.csv"
        evoprotgrad_df.to_csv(filepath, index=False)
        logger.info(f"Saved EvoProt Grad generation {generation + 1} results to {filepath}")
        
        # Print some stats about this generation
        logger.info(f"Generation {generation + 1} stats:")
        logger.info(f"  Best predicted PAE Interaction: {evoprotgrad_df['pred_pae_interaction'].min():.4f}")
        logger.info(f"  Median predicted PAE Interaction: {evoprotgrad_df['pred_pae_interaction'].median():.4f}")
        logger.info(f"  Worst predicted PAE Interaction: {evoprotgrad_df['pred_pae_interaction'].max():.4f}")
        
        # Select sequences to fold
        already_folded = get_folded_ids()
        logger.info(f"Total N folded = {len(already_folded)} sequences")
        evoprotgrad_df = evoprotgrad_df[~evoprotgrad_df['seq_id'].isin(already_folded)]
        logger.info(f"N evoprotgrad_df after filtering = {len(evoprotgrad_df)}")
        seqs_to_fold = evoprotgrad_df.head(n_evoprotgrad_survivors).set_index('seq_id')['binder_sequence'].to_dict()

        logger.info(f"Folding {len(seqs_to_fold)} sequences")
        batch_size = int(n_evoprotgrad_survivors / COLABFOLD_GPU_CONCURRENCY_LIMIT)
        batch_size = max(batch_size, 1)
        fold_results = parallel_fold_and_extract.remote(
            binder_sequences=seqs_to_fold,
            template_a3m_path=None,
            target_seq=None,
            batch_size=batch_size,
            num_models=1, 
            num_recycle=1)
        
        fold_df = pd.DataFrame(fold_results)
        fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
        fold_df = fold_df.sort_values(by='pae_interaction', ascending=True).reset_index(drop=True)
        fold_df['generation'] = generation + 1
        
        # Save the fold results
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
        fold_df.to_csv(fp, index=False)
        logger.info(f"Saved fold results to {fp}")
        
        if select_from_all_folded:
            fold_df = pd.concat([fold_df, get_fold_results()])
        fold_df = fold_df.drop_duplicates(subset=['seq_id']).sort_values(by='pae_interaction', ascending=True)
        
        # Log statistics for pae_interaction
        logger.info(f"  Best (lowest) PAE Interaction: {fold_df['pae_interaction'].min():.4f}")
        
        # Get the 2nd, 5th, and 10th best PAE Interactions
        second_best_pae = fold_df['pae_interaction'].nsmallest(2).iloc[-1]
        fifth_best_pae = fold_df['pae_interaction'].nsmallest(5).iloc[-1]
        tenth_best_pae = fold_df['pae_interaction'].nsmallest(10).iloc[-1]
        logger.info(f"  2nd Best PAE Interaction: {second_best_pae:.4f}")
        logger.info(f"  5th Best PAE Interaction: {fifth_best_pae:.4f}")
        logger.info(f"  10th Best PAE Interaction: {tenth_best_pae:.4f}")
        
        if folded_top_k is not None:
            fold_df = fold_df.head(folded_top_k)
            # sample based on pae_interaction
            p = fold_df['pae_interaction'].values / fold_df['pae_interaction'].sum()
            fold_df = fold_df.sample(n=n_colabfold_survivors, replace=False, weights=p)
            
        next_generation = fold_df.head(n_colabfold_survivors)['binder_sequence'].tolist()
        
        # Update current_generation for the next iteration
        current_generation = next_generation
        logger.info(f"Selected {len(current_generation)} sequences for the next generation")
        
        # Log sequences for next generation
        logger.info("Sequences selected for next generation:")
        for seq in current_generation:
            logger.info(f"{seq}")
        
        mut_strs = []
        for seq in current_generation:
            try:
                mut_str = get_mutation_diff(seq, EGFS)
                mut_strs.append(mut_str)
            except Exception as e:
                logger.warning(f"Failed to calculate mut_str for sequence: {seq}. Error: {str(e)}")
                mut_strs.append(None)
        try:
            for mut_str in mut_strs:
                logger.info(f"{mut_str}")
        except Exception as e:
            logger.error(f"Failed to log mut_strs. Error: {str(e)}")
    
    logger.info("Evolution complete.")
    return fold_df
