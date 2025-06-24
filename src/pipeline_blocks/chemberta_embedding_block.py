import torch
import sys
import os
import pandas as pd
import time
from tqdm import tqdm
import glob

# Add the src directory to PYTHONPATH at runtime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.chemberta_embedding import ChembertaModel

class ChembertaEmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self):
        self.ligands = None
        self.ligands_embd = None


    def set_input(self, ligands):
        """
        Set the input ligands.
        """
        self.ligands = ligands


    def get_output(self):
        """
        Get the processed ligands.
        """
        return self.ligands_embd
    
    
    def run(self, batch_size: int = 32, save_intermediate: bool = True, output_dir: str = 'data/embeddings'):
        """
        Run the embedding step for the provided ligands.
        
        Args:
            batch_size (int): Number of sequences to process at once. Default is 32.
            save_intermediate (bool): Whether to save intermediate results to disk.
            output_dir (str): Directory to save intermediate results.
        """
        if self.ligands is None:
            raise ValueError("Ligands input not set. Use set_input() before calling run().")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Processing {len(self.ligands)} ligands on {device}")
        
        # Create output directory if it doesn't exist
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
            # Clean up any existing temporary files
            temp_files = glob.glob(os.path.join(output_dir, 'temp_ligand_embeddings_*.pkl'))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    print(f"Removed old temporary file: {os.path.basename(temp_file)}")
                except OSError:
                    pass
        
        embedder = ChembertaModel(device=device)
        
        # Process embeddings and convert to numpy arrays immediately to save memory
        all_embeddings = []
        smiles_processed = []
        
        chunk_size = batch_size * 20  # Larger chunks for better efficiency
        total_chunks = (len(self.ligands) + chunk_size - 1) // chunk_size
        
        start_time = time.time()
        last_intermediate_file = None
        
        # Create progress bar for chunks
        with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
            for chunk_idx in range(total_chunks):
                i = chunk_idx * chunk_size
                chunk_end = min(i + chunk_size, len(self.ligands))
                chunk_ligands = self.ligands[i:chunk_end]
                
                chunk_start_time = time.time()
                
                # Get embeddings for this chunk
                chunk_embeddings = embedder.embed(chunk_ligands, batch_size=batch_size, show_progress=True)
                
                # Convert to numpy arrays and store
                chunk_arrays = [emb.cpu().numpy() for emb in chunk_embeddings]
                all_embeddings.extend(chunk_arrays)
                smiles_processed.extend(chunk_ligands)
                
                chunk_time = time.time() - chunk_start_time
                elapsed_total = time.time() - start_time
                
                # Update progress bar with timing information
                remaining_chunks = total_chunks - (chunk_idx + 1)
                avg_chunk_time = elapsed_total / (chunk_idx + 1)
                estimated_remaining = remaining_chunks * avg_chunk_time
                
                pbar.set_postfix({
                    'Ligands': f"{len(smiles_processed)}/{len(self.ligands)}",
                    'Elapsed': f"{elapsed_total:.1f}s",
                    'ETA': f"{estimated_remaining:.1f}s",
                    'Chunk_time': f"{chunk_time:.1f}s"
                })
                pbar.update(1)
                
                # Save intermediate results to disk and delete previous
                if save_intermediate and (chunk_end % (batch_size * 100) == 0 or chunk_end == len(self.ligands)):
                    # Delete previous intermediate file
                    if last_intermediate_file and os.path.exists(last_intermediate_file):
                        try:
                            os.remove(last_intermediate_file)
                        except OSError:
                            pass
                    
                    # Save new intermediate file
                    intermediate_filename = f'temp_ligand_embeddings_{len(smiles_processed)}.pkl'
                    intermediate_path = os.path.join(output_dir, intermediate_filename)
                    
                    temp_df = pd.DataFrame({
                        'smiles': smiles_processed,
                        'embedding': all_embeddings
                    })
                    temp_df.to_pickle(intermediate_path)
                    last_intermediate_file = intermediate_path
                    
                    # disabled since it breaks progress bar
                    #print(f"\nSaved intermediate results: {len(smiles_processed)} ligands to {intermediate_filename}")
        
        # Create final DataFrame
        self.ligands_embd = pd.DataFrame({
            'smiles': smiles_processed,
            'embedding': all_embeddings
        })
        
        total_time = time.time() - start_time
        print(f"\nCompleted processing {len(self.ligands)} ligands in {total_time:.1f} seconds ({total_time/60:.1f} minutes).")
        
        # Clean up the last intermediate file since we have the final result
        if last_intermediate_file and os.path.exists(last_intermediate_file):
            try:
                os.remove(last_intermediate_file)
                print(f"Cleaned up intermediate file: {os.path.basename(last_intermediate_file)}")
            except OSError:
                pass
        