
import os
from modal import App, Secret, gpu, Image, enter, method
import logging
from datetime import datetime
from binder_design import DATA_DIR, EGFS, EGFR
from binder_design.utils import get_mutation_diff, hash_seq
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def download_models():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    
    logger.info("Starting model download")
    try:
        snapshot_download(repo_id="ChatterjeeLab/PepMLM-650M", repo_type="model", token=os.environ["HUGGINGFACE_TOKEN"])
        logger.info("Model download completed successfully")
        move_cache()
        logger.info("Cache moved successfully")
    except Exception as e:
        logger.error(f"Error during model download or cache move: {str(e)}")
        raise
    
image = (
    Image
    .debian_slim(python_version="3.10")
    .pip_install('uv')
    .pip_install('Bio')
    .pip_install('transformers')
    .run_commands("uv pip install  --system --compile-bytecode torch huggingface_hub pandas tqdm datasets", gpu="a10g")
    .run_function(download_models, secrets=[Secret.from_dotenv()])
    )

app = App(name="pepmlm", image=image)

with image.imports():
    import torch
    import transformers
    import huggingface_hub
    import pandas
    import tqdm
    
    import numpy as np
    import pandas as pd
    
  
@app.cls(
    container_idle_timeout=150,
    image=image,
    secrets=[Secret.from_dotenv()],
    gpu="a10g",
    concurrency_limit=20,
    timeout=9600,
)
class PepMLM:

    @enter()
    def enter(self):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        logger.info("Initializing PepMLM")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info("Loading model and tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("ChatterjeeLab/PepMLM-650M")
        self.model = AutoModelForMaskedLM.from_pretrained("ChatterjeeLab/PepMLM-650M", token=os.environ["HUGGINGFACE_TOKEN"]).to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
        
    def _compute_ppl(self, target_seq, binder_seq):
        import numpy as np
        import time
        '''
        For alternative computation of PPL (in batch/matrix format), please check our github repo:
        https://github.com/programmablebio/pepmlm/blob/main/scripts/generation.py
        '''
        start_time = time.time()
        logger.debug(f"Computing pseudo-perplexity for target sequence: {target_seq[:10]}... and binder sequence: {binder_seq}")
        sequence = target_seq + binder_seq
        tensor_input = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        total_loss = 0

        # Loop through each token in the binder sequence
        for i in range(-len(binder_seq)-1, -1):
            # Create a copy of the original tensor
            masked_input = tensor_input.clone()

            # Mask one token at a time
            masked_input[0, i] = self.tokenizer.mask_token_id
            # Create labels
            labels = torch.full(tensor_input.shape, -100).to(self.device)
            labels[0, i] = tensor_input[0, i]

            # Get model prediction and loss
            with torch.no_grad():
                outputs = self.model(masked_input, labels=labels)
                total_loss += outputs.loss.item()

        # Calculate the average loss
        avg_loss = total_loss / len(binder_seq)

        # Calculate pseudo perplexity
        pseudo_perplexity = np.exp(avg_loss)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Computed pseudo-perplexity: {pseudo_perplexity}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        return pseudo_perplexity
    
    

    def _compute_ppl_vectorized(self, target_seq, binder_seq):
        import torch
        import numpy as np
        import time

        start_time = time.time()
        logger.info(f"Computing vectorized pseudo-perplexity for target sequence: {target_seq[:10]}... and binder sequence: {binder_seq}")
        sequence = target_seq + binder_seq
        original_input = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        length_of_binder = len(binder_seq)

        # Prepare a batch with each row having one masked token from the binder sequence
        masked_inputs = original_input.repeat(length_of_binder, 1)
        positions_to_mask = torch.arange(-length_of_binder - 1, -1, device=self.device)

        masked_inputs[torch.arange(length_of_binder), positions_to_mask] = self.tokenizer.mask_token_id

        # Prepare labels for the masked tokens
        labels = torch.full_like(masked_inputs, -100)
        labels[torch.arange(length_of_binder), positions_to_mask] = original_input[0, positions_to_mask]

        # Process in batches to avoid memory issues
        batch_size = 32  # Adjust this based on your GPU memory
        total_loss = 0

        for i in range(0, length_of_binder, batch_size):
            batch_masked_inputs = masked_inputs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Get model predictions and calculate loss
            with torch.no_grad():
                outputs = self.model(batch_masked_inputs, labels=batch_labels)
                total_loss += outputs.loss.item() * len(batch_masked_inputs)

        # Calculate average loss
        avg_loss = total_loss / length_of_binder
        pseudo_perplexity = np.exp(avg_loss)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Computed vectorized pseudo-perplexity: {pseudo_perplexity}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        
        return pseudo_perplexity
    
    def _compute_ppl_unmasked(self, target_seq, binder_seq):
        '''
        Computes pseudo-perplexity without masking each residue.
        '''
        sequence = target_seq + binder_seq
        tensor_input = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        labels = torch.full(tensor_input.shape, -100).to(self.device)
        labels[0, -len(binder_seq):] = tensor_input[0, -len(binder_seq):]  # Set labels for binder tokens

        with torch.no_grad():
            outputs = self.model(tensor_input, labels=labels)
            total_loss = outputs.loss.item()

        # Calculate pseudo perplexity
        pseudo_perplexity = np.exp(total_loss / len(binder_seq))
        return pseudo_perplexity
        
    @method()
    def compute_pseudo_perplexity(self, target_seq, binder_seq):
        return self._compute_ppl_unmasked(target_seq, binder_seq)
        
    def generate_peptide_for_single_sequence(self, protein_seq, peptide_length=15, top_k=3, num_binders=4):
        from torch.distributions.categorical import Categorical
        logger.info(f"Generating peptides for sequence: {protein_seq[:10]}...")
        peptide_length = int(peptide_length)
        top_k = int(top_k)
        num_binders = int(num_binders)

        binders_with_ppl = []

        for i in range(num_binders):
            logger.info(f"Generating binder {i+1}/{num_binders}")
            # Generate binder
            masked_peptide = '<mask>' * peptide_length
            input_sequence = protein_seq + masked_peptide
            inputs = self.tokenizer(input_sequence, return_tensors="pt").to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
            mask_token_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            logits_at_masks = logits[0, mask_token_indices]

            # Restrict to allowed tokens
            allowed_indices = torch.tensor(range(4, 24)).to(self.device)
            logits_at_masks[:, torch.tensor([i for i in range(logits_at_masks.shape[1]) if i not in allowed_indices])] = float('-inf')

            # Apply top-k sampling
            top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            predicted_indices = Categorical(probabilities).sample()
            predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)

            generated_binder = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True).replace(' ', '')
            logger.debug(f"Generated binder: {generated_binder}")

            # Compute PPL for the generated binder
            ppl_value = self._compute_ppl_unmasked(protein_seq, generated_binder)

            # Add the generated binder and its PPL to the results list
            binders_with_ppl.append([generated_binder, ppl_value])

        logger.info(f"Generated {num_binders} binders for the sequence")
        return binders_with_ppl

    @method()
    def generate_peptide(self, input_seqs, peptide_length=15, top_k=3, num_binders=4):
        import pandas as pd
        logger.info("Starting peptide generation")
        if isinstance(input_seqs, str):  # Single sequence
            logger.info("Processing single sequence")
            binders = self.generate_peptide_for_single_sequence(input_seqs, peptide_length, top_k, num_binders)
            result = pd.DataFrame(binders, columns=['Binder', 'Pseudo Perplexity'])
            logger.info("Peptide generation completed for single sequence")
            return result

        elif isinstance(input_seqs, list):  # List of sequences
            logger.info(f"Processing list of {len(input_seqs)} sequences")
            results = []
            for i, seq in enumerate(input_seqs):
                logger.info(f"Processing sequence {i+1}/{len(input_seqs)}")
                binders = self.generate_peptide_for_single_sequence(seq, peptide_length, top_k, num_binders)
                for binder, ppl in binders:
                    results.append([seq, binder, ppl])
            result = pd.DataFrame(results, columns=['Input Sequence', 'Binder', 'Pseudo Perplexity'])
            logger.info("Peptide generation completed for all sequences")
            return result
        
    @method()
    def edit_binders(self, target_seq, binder_seqs, frac_residues_to_mask=0.1, top_k=3, num_variations=10):
        from torch.distributions.categorical import Categorical
        import numpy as np
        import pandas as pd
        
        if isinstance(binder_seqs, str):
            binder_seqs = [binder_seqs]
        
        all_mutated_binders = []
        
        for binder_seq in binder_seqs:
            logger.info(f"Mutating binder: {binder_seq}")
            binder_length = len(binder_seq)
            num_residues_to_mask = max(1, int(binder_length * frac_residues_to_mask))
            
            # Calculate perplexity for the original binder sequence
            original_ppl = self._compute_ppl_unmasked(target_seq, binder_seq)
            
            mutated_binders = [{
                'target_seq': target_seq,
                'parent_binder': binder_seq,
                'binder': binder_seq,
                'unmasked_ppl': original_ppl,
                'mask_positions': '',
                'mutation': '',
                'seq_id': hash_seq(binder_seq)
            }]
            
            for _ in range(num_variations):
                # Randomly select positions to mask
                mask_positions = np.random.choice(binder_length, num_residues_to_mask, replace=False)
                
                # Create masked binder sequence
                masked_binder = list(binder_seq)
                for pos in mask_positions:
                    masked_binder[pos] = self.tokenizer.mask_token
                masked_binder = ''.join(masked_binder)
                
                # Prepare input for the model
                input_sequence = target_seq + masked_binder
                inputs = self.tokenizer(input_sequence, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                # Get logits for masked positions
                mask_token_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                logits_at_masks = logits[0, mask_token_indices]
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = logits_at_masks.topk(top_k, dim=-1)
                probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
                predicted_indices = Categorical(probabilities).sample()
                predicted_token_ids = top_k_indices.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)
                
                # Replace masked tokens with predicted ones
                mutated_binder = list(binder_seq)
                for pos, token_id in zip(mask_positions, predicted_token_ids):
                    mutated_binder[pos] = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                mutated_binder = ''.join(mutated_binder).replace(' ', '')
                
                # Compute PPL for the mutated binder
                ppl_value = self._compute_ppl_unmasked(target_seq, mutated_binder)
                
                mutated_binders.append({
                    'target_seq': target_seq,
                    'parent_binder': binder_seq,
                    'binder': mutated_binder,
                    'unmasked_ppl': ppl_value,
                    'mask_positions': ','.join(map(str, mask_positions)),
                    'mutation': get_mutation_diff(binder_seq, mutated_binder),
                    'seq_id': hash_seq(mutated_binder)
                })
            
            all_mutated_binders.extend(mutated_binders)
            logger.info(f"Generated {num_variations} mutations for binder: {binder_seq}")
        
        logger.info(f"Generated mutations for {len(binder_seqs)} binders")
        result = pd.DataFrame(all_mutated_binders)
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result
        



@app.local_entrypoint()
def test():
    
    pepmlm = PepMLM()
    
    # Test compute_pseudo_perplexity
    target_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    binder_seq = "ACDEFGHIKLMNPQRS"
    ppl = pepmlm.compute_pseudo_perplexity.remote(target_seq, binder_seq)
    print(f"Pseudo-perplexity for the given sequence: {ppl}")

    # Test generate_peptide with a single sequence
    single_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    single_result = pepmlm.generate_peptide.remote(single_seq, peptide_length=15, top_k=3, num_binders=2)
    print("\nGenerated peptides for single sequence:")
    print(single_result)

    # Test edit_binder
    target_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    binder_seq = "ACDEFGHIKLMNPQRS"
    mutated_binders = pepmlm.edit_binders.remote(target_seq, binder_seq, frac_residues_to_mask=0.1, top_k=3, num_variations=10)
    print("\nMutated binders:")
    print(mutated_binders)
    
@app.local_entrypoint()
def test_egfs_edit():
    import pandas as pd
    pepmlm = PepMLM()
    target_seq = EGFR
    binder_seq = EGFS
    mutated_binders = pepmlm.edit_binders.remote(target_seq, binder_seq, frac_residues_to_mask=0.15, top_k=5, num_variations=20)
    
    # Calculate perplexity for the original binder sequence
    original_ppl = pepmlm.compute_pseudo_perplexity.remote(target_seq, binder_seq)
    
    # Add the original binder and its perplexity to the DataFrame
    original_binder = pd.DataFrame({
        'target_seq': [target_seq],
        'parent_binder': [binder_seq],
        'binder': [binder_seq],
        'unmasked_ppl': [original_ppl],
        'mask_positions': [''],
        'mutation': [''],
        'seq_id': [hash_seq(binder_seq)]
        
    })
    
    # Concatenate the original binder with the mutated binders
    mutated_binders = pd.concat([original_binder, mutated_binders], ignore_index=True)
    
    mutated_binders = mutated_binders.sort_values(by='ppl')
    print("\nMutated binders:")
    print(mutated_binders)
    
    
@app.function(timeout=4800)
def parallel_edit_binders(binder_seqs, target_seq, frac_residues_to_mask=0.15, top_k=5, num_variations_per_binder=20):
    pepmlm = PepMLM()
    
    # Prepare batches
    batches = [(target_seq, binder, frac_residues_to_mask, top_k, num_variations_per_binder) for binder in binder_seqs]

    # Run edit_binder in parallel
    all_mutated_binders = pepmlm.edit_binders.starmap(batches)
    
    # Combine results
    combined_results = pd.concat(all_mutated_binders, ignore_index=True)
    
    return combined_results

# @app.function(timeout=12800)
# def edit_binders_parallel(binder_seqs=None, target_seq=EGFR, frac_residues_to_mask=0.05, top_k=5, num_variations_per_binder=10, min_n_binder_seqs=2):
    
#     if binder_seqs is None:
#         binder_seqs = [EGFS]
    
#     if len(binder_seqs) < min_n_binder_seqs:
#         # Multiply binder_seqs to meet the minimum requirement
#         multiplier = -(-min_n_binder_seqs // len(binder_seqs))  # Ceiling division
#         binder_seqs = binder_seqs * multiplier
        
#         print(f"Expanded binder sequences from {len(binder_seqs) // multiplier} to {len(binder_seqs)} to meet minimum requirement.")
#     results = parallel_edit_binders.remote(binder_seqs, target_seq, frac_residues_to_mask, top_k, num_variations_per_binder)
    
#     # Sort the results by perplexity
#     sorted_results = results.sort_values(by='ppl')
    
    
#     return sorted_results

@app.local_entrypoint()
def evolve_binders(
    init_binder_seqs=None, 
    target_seq=EGFR, 
    frac_residues_to_mask=0.025, 
    top_k=8, 
    num_variations_per_binder=100, 
    min_n_binder_seqs=20,
    n_generations=40,
    n_survivors=50,
):
    if init_binder_seqs is None:
        init_binder_seqs = [EGFS]
    
    # Ensure we have at least min_n_binder_seqs to start with
    if len(init_binder_seqs) < min_n_binder_seqs:
        multiplier = -(-min_n_binder_seqs // len(init_binder_seqs))  # Ceiling division
        init_binder_seqs = init_binder_seqs * multiplier
        print(f"Expanded initial binder sequences from {len(init_binder_seqs) // multiplier} to {len(init_binder_seqs)} to meet minimum requirement.")

    current_generation = init_binder_seqs
    
    for generation in range(n_generations):
        print(f"Starting generation {generation + 1}/{n_generations}")
        
        # Generate variations for the current generation
        results = parallel_edit_binders.remote(current_generation, target_seq, frac_residues_to_mask, top_k, num_variations_per_binder)
        
        # Sort the results by perplexity (lower is better)
        sorted_results = results.sort_values(by='unmasked_ppl')
        
        # Select the top n_survivors as the next generation
        next_generation = sorted_results['binder'].head(n_survivors).tolist()
        
        # Save this generation's results
        save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = DATA_DIR / f"pepmlm_binders_gen{generation+1}_{save_time}.csv"
        sorted_results.to_csv(filepath, index=False)
        print(f"Saved generation {generation + 1} results to {filepath}")
        
        # Print some stats about this generation
        print(f"Generation {generation + 1} stats:")
        print(f"  Best PPL: {sorted_results['unmasked_ppl'].min()}")
        print(f"  Median PPL: {sorted_results['unmasked_ppl'].median()}")
        print(f"  Worst PPL: {sorted_results['unmasked_ppl'].max()}")
        
        # Update current_generation for the next iteration
        current_generation = next_generation
    
    print("Evolution complete.")
    return sorted_results
    