import os
import argparse
import tempfile
import shutil
import subprocess
import logging
from binder_design import TEMPLATE_A3M_PATH, EGFS, EGFR, COLABFOLD_GPU_CONCURRENCY_LIMIT, FOLD_RESULTS_DIR
from modal import Image, App, method, enter, Dict
from binder_design.utils import get_mutation_diff, hash_seq
import io
import zipfile
import re
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = App("colabfold")


image = (
    Image
    # .from_registry("nvidia/cuda:12.4.99-runtime-ubuntu22.04", add_python="3.11")
    .debian_slim(python_version="3.11")
    .micromamba(python_version="3.11")
    .apt_install("wget", "git", "curl")
    # .pip_install(
    #     "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold",
    #     gpu="a100",
    # )
    # .micromamba_install(
    #     "kalign2=2.04", "hhsuite=3.3.0", "pdbfixer", channels=["conda-forge", "bioconda"],
    #     gpu="a100",
    # )
    # .run_commands(
    #     'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
    #     gpu="a100",
    # )
    .run_commands('wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh')
    .run_commands('bash install_colabbatch_linux.sh', gpu="a100",)
    .pip_install('biopython')
    .pip_install('pandas')
    # .pip_install("grpclib")
    # .run_commands('export PATH="/localcolabfold/colabfold-conda/bin:$PATH"')
)


def generate_a3m_files(binder_sequences, output_folder, template_a3m_path=TEMPLATE_A3M_PATH, target_sequence=None):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(template_a3m_path, 'r') as template_file:
        template_lines = template_file.readlines()
    
    # Extract the target sequence from the template if target_sequence is None
    if target_sequence is None:
        template_sequence = template_lines[2].strip()
        binder_length = int(template_lines[0].split(',')[0].strip('#'))
        target_sequence = template_sequence[binder_length:]
    
    for name, binder_seq in binder_sequences.items():
        output_path = os.path.join(output_folder, f"{name}.a3m")
        
        with open(output_path, 'w') as output_file:
            # Write the header lines
            output_file.writelines(template_lines[:2])
            
            # Write the concatenated sequences
            output_file.write(f"{binder_seq}{target_sequence}\n")
            
            # Write the rest of the template file
            output_file.writelines(template_lines[3:])
    
    return output_folder


# Manual three-letter to one-letter amino acid code conversion
aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def three_to_one(three_letter_code):
    return aa_dict.get(three_letter_code, 'X')







with image.imports():
    from Bio import PDB
    import numpy as np
    import pandas as pd
    

@app.cls(image=image, gpu='a100', timeout=9600, concurrency_limit=COLABFOLD_GPU_CONCURRENCY_LIMIT)
class LocalColabFold:
    @enter()
    def setup(self):
        from Bio import PDB
        import numpy as np
        import pandas as pd
        
        # Set up the environment when the container starts
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def fold(self, binder_sequences, template_a3m_path, target_sequence=None,  **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # A3M-based approach
            input_path = generate_a3m_files(
                binder_sequences=binder_sequences,
                output_folder=temp_dir,
                template_a3m_path=template_a3m_path,
                target_sequence=target_sequence
            )
            logger.info(f"Generated A3M files in: {input_path}")

            out_dir = "output"
            os.makedirs(out_dir, exist_ok=True)
            logger.info(f"Created output directory: {out_dir}")

            cmd = ["colabfold_batch", input_path, out_dir]
            
            # Handle arguments
            for key, value in kwargs.items():
                key = key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                elif value is not None:
                    cmd.extend([f"--{key}", str(value)])
                    
            # hardcode zip
            cmd.append('--zip')
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Command output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with error: {e}")
                logger.error(f"Error output: {e.stderr}")
                raise
            
            logger.info(f'input directory contents: {os.listdir(input_path)}')
            logger.info(f"Output directory contents: {os.listdir(out_dir)}")
            logger.info(f'current directory contents: {os.listdir(".")}')
            
            # Find and return the zip result
            try:
                zip_file = next(f for f in os.listdir(out_dir) if f.endswith(".zip"))
                zip_path = os.path.join(out_dir, zip_file)
                logger.info(f"Found zip file: {zip_path}")
                
                # Extract metrics and PDBs here
                all_results = self.extract_metrics_and_pdbs(zip_path, out_dir)
                
                logger.info(f"Extracted {len(all_results)} results")
                
                # return {
                #     'zip_content': open(zip_path, 'rb').read(),
                #     'results': all_results,
                # }
                return all_results
            except StopIteration:
                logger.error(f"No zip file found in {out_dir}")
                logger.error(f"Directory contents: {os.listdir(out_dir)}")
                raise FileNotFoundError(f"No zip file found in {out_dir}")
            
    @staticmethod
    def extract_sequence_from_pdb(pdb_content):
        parser = PDB.PDBParser()
        structure = parser.get_structure("protein", io.StringIO(pdb_content.decode('utf-8')))
        
        sequences = {'A': '', 'B': ''}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id in sequences:
                    for residue in chain:
                        if PDB.is_aa(residue):
                            sequences[chain_id] += three_to_one(residue.resname)
        
        return {
            'binder': sequences['A'],
            'target': sequences['B'],
            'binder_length': len(sequences['A']),
            'target_length': len(sequences['B']),
        }
        
    @staticmethod
    def extract_sequences(zip_ref, seq_name):
        pdb_file = next(name for name in zip_ref.namelist() if name.startswith(seq_name) and name.endswith('.pdb'))
        with zip_ref.open(pdb_file) as file:
            pdb_content = file.read()
            sequences = LocalColabFold.extract_sequence_from_pdb(pdb_content)
        return sequences
    
    @staticmethod
    def extract_scores(zip_ref, seq_name, binder_length):
        pattern = r'model_(\d+)'
        score_jsons = [name for name in zip_ref.namelist() if seq_name in name and '_scores_' in name]
        
        results = []
        for json_file in score_jsons:
            match = re.search(pattern, json_file)
            if match:
                model_number = int(match.group(1))
                
            with zip_ref.open(json_file) as file:
                data = json.load(file)
                
                plddt_array = np.array(data['plddt'])
                pae_array = np.array(data['pae'])
                
                pae_interaction = (pae_array[binder_length:, :binder_length].mean() + pae_array[:binder_length, binder_length:].mean()) / 2
                binder_plddt = plddt_array[:binder_length].mean()
                binder_pae = pae_array[:binder_length, :binder_length].mean()
                
                result = {
                    'model_number': model_number,
                    'binder_plddt': float(binder_plddt),
                    'binder_pae': float(binder_pae),
                    'pae_interaction': float(pae_interaction),
                    'ptm': data['ptm'],
                }
            results.append(result)
        
        return results
    
    @staticmethod
    def extract_metrics_and_pdbs(zip_path, output_dir):
        logger.info(f"Extracting metrics and PDBs from {zip_path}")
        all_results = []
        pdbs = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdb_files = [name for name in zip_ref.namelist() if name.endswith('.pdb')]
            seq_names = list(set([n.split('_unrelaxed_rank')[0] for n in pdb_files]))
            
            logger.info(f"Found {len(seq_names)} sequence names in the zip file")
            
            for seq_name in seq_names:
                sequences = LocalColabFold.extract_sequences(zip_ref, seq_name)
                binder_length = sequences['binder_length']
                target_length = sequences['target_length']
                
                logger.info(f"Extracted sequences for {seq_name}: binder length {binder_length}, target length {target_length}")
                
                scores = LocalColabFold.extract_scores(zip_ref, seq_name, binder_length)
                
                logger.info(f"Extracted {len(scores)} scores for {seq_name}")
                
                for score in scores:
                    result = {
                        'seq_name': seq_name,
                        'binder_sequence': sequences['binder'],
                        'target_sequence': sequences['target'],
                        'binder_length': binder_length,
                        'target_length': target_length,
                        'model_number': score['model_number'],
                        'binder_plddt': score['binder_plddt'],
                        'binder_pae': score['binder_pae'],
                        'pae_interaction': score['pae_interaction'],
                        'ptm': score['ptm'],
                        'seq_id': hash_seq(sequences['binder']),
                        # 'mut_str': get_mutation_diff(sequences['binder'], EGFS),
                    }
                    all_results.append(result)
                    
                    # Find and extract PDB files
                    pdb_files = [name for name in zip_ref.namelist() if name.endswith('.pdb') and seq_name in name and f"rank_{score['model_number']:03d}" in name]
                    for pdb_filename in pdb_files:
                        with zip_ref.open(pdb_filename) as pdb_file:
                            pdb_content = pdb_file.read().decode('utf-8')
                            pdbs[f"{seq_name}_model_{score['model_number']}"] = pdb_content
                    
                    logger.info(f"Extracted PDB content for {seq_name}_model_{score['model_number']}")
                    
                    result['pdb_content'] = pdb_content
        
        logger.info(f"Extracted {len(all_results)} total results and {len(pdbs)} PDB files")
        return all_results
        

    
@app.function(timeout=4800)
def fold_and_extract(
    binder_sequences: dict, 
    template_a3m_path: str=TEMPLATE_A3M_PATH, 
    target_sequence: str = None, 
    zip_results: bool = True,
    **kwargs):
    lcf = LocalColabFold()
    result = lcf.fold.remote(
        binder_sequences=binder_sequences,
        template_a3m_path=template_a3m_path,
        target_sequence=target_sequence,
        zip=zip_results,
        **kwargs
    )
    return result['results']

@app.function(timeout=12800)
def parallel_fold_and_extract(
    binder_sequences: dict, 
    template_a3m_path: str = None, 
    target_sequence: str = None, 
    batch_size: int = 10,
    **kwargs):
    
    if template_a3m_path is None:
        template_a3m_path = TEMPLATE_A3M_PATH
    
    lcf = LocalColabFold()
    all_results = []
    
    # Prepare batches
    batches = []
    for i in range(0, len(binder_sequences), batch_size):
        batch = dict(list(binder_sequences.items())[i:i+batch_size])
        batches.append((batch, template_a3m_path, target_sequence))


    all_results = []
    for result in lcf.fold.starmap(batches, kwargs=kwargs):
        all_results.extend(result)
    
    return all_results
    
    
@app.local_entrypoint()
def test():
    # Test sequence-based folding
    sequences = {
        'binder': 'NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW',
        'target': 'EGFR_SEQUENCE_HERE'
    }
    results_seq = fold_and_extract.remote(sequences=sequences)

    # Test a3m-based folding with provided target sequence
    template_a3m_path = TEMPLATE_A3M_PATH
    binder_sequences = {
        "binder1": EGFS,
    }
    target_sequence = EGFR
    results_a3m_with_target =  fold_and_extract.remote(
        template_a3m_path=template_a3m_path,
        binder_sequences=binder_sequences,
        target_sequence=target_sequence
    )

    # Test a3m-based folding without provided target sequence
    results_a3m_without_target = fold_and_extract.remote(
        template_a3m_path=template_a3m_path,
        binder_sequences=binder_sequences
    )
    
@app.local_entrypoint()
def manual_parallel_fold():
    def mutate_seq(seq, position, new_aa):
        return seq[:position-1] + new_aa + seq[position:]

    def apply_mutation(seq, mutation_str):
        new_aa = mutation_str[-1]
        position = int(mutation_str[:-1])
        return mutate_seq(seq, position, new_aa)

    egfs = 'NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW'
    seed = egfs
    seed = 'PSYSGCPSSYDGYCGNGGVCMHIESLDSYTCQCVIGYSGDRVQTRDLRWT'
    seed = 'ISYSACPLSYDGVCGNGGVCKHALSLDSYTCQCVWGYSGDRVQTRDLRYT'

    mutations = [
        # '5A',
        # '6A',
        
        ]
    for pos in range(7, 50):
        mutations.append(f'{pos}A')
    seqs = {}
    for mutation in mutations:
        seq = apply_mutation(seed, mutation)
        seqs[hash_seq(seq)] = seq
        
    results = list(parallel_fold_and_extract.remote(binder_sequences=seqs, batch_size=3, num_recycle=1, num_models=1))
    fold_df = pd.DataFrame(results)
    fold_df['mut_str'] = fold_df['binder_sequence'].apply(get_mutation_diff, seq2=EGFS)
    fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
    # save the fold_df
    fold_df.to_csv(fp, index=False)
    
@app.local_entrypoint()
def manual_parallel_fold_validate():
    seed = 'ISYSACPLSYDGVCGNGGVCKHALSLDSYTCQCVWGYSGDRVQTRDLRYT'

    seqs = {}
    seqs[hash_seq(seed)] = seed
        
        
    results = list(parallel_fold_and_extract.remote(binder_sequences=seqs, batch_size=3, num_recycle=3, num_models=5))
    fold_df = pd.DataFrame(results)
    fold_df['mut_str'] = fold_df['binder_sequence'].apply(get_mutation_diff, seq2=EGFS)
    fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
    # save the fold_df
    fold_df.to_csv(fp, index=False)