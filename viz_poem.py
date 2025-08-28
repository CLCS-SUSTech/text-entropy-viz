# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from viz import nll_to_rgb
# %%
def run_poem_viz_doc(doc_name, text_dir, nll_dir, tokenizer, output_dir='.', output_file=None, global_nll_min=None, global_nll_max=None, masked_positions=None):
    """
    Note: There are multiple lines in poem text file, while only one line in nll file
    
    Args:
        doc_name: document name
        text_dir: directory containing text files
        nll_dir: directory containing NLL files
        tokenizer: tokenizer object
        output_dir: output directory for HTML files
        global_nll_min: float, optional global minimum NLL value for normalization
        global_nll_max: float, optional global maximum NLL value for normalization
        masked_positions: list of int, optional positions to mask in the text, which will not be colored
    """
    text_file = os.path.join(text_dir, f"{doc_name}.txt")
    nll_file = os.path.join(nll_dir, f"{doc_name}.txt")
    if not os.path.exists(text_file):
        print(f"Error: Text file not found: {text_file}")
        return
    if not os.path.exists(nll_file):
        print(f"Error: NLL file not found: {nll_file}")
        return 
    
    # Read text and NLL data
    with open(text_file, 'r', encoding='utf-8') as f:
        text_content = f.read().strip()
    with open(nll_file, 'r', encoding='utf-8') as f:
        nll_content = f.read().strip()
    all_nlls = list(map(float, nll_content.split()))
    
    if global_nll_min is None:
        global_nll_min = min(all_nlls) 
    if global_nll_max is None:
        global_nll_max = max(all_nlls)
    print(f"Global NLL range: {global_nll_min:.4f} to {global_nll_max:.4f}") 

    # Create combined HTML for all lines
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>NLL Visualization - {doc_name}</title>
        <style>
            body {{ font-family: 'Courier New', monospace; font-size: 16px; line-height: 1.6; margin: 20px; }}
            .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; }}
            .legend {{ margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
            .legend-item {{ display: inline-block; margin-right: 20px; }}
            .color-box {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; border: 1px solid #ccc; }}
            .sentence {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .sentence-number {{ font-weight: bold; color: #666; margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h2>NLL Visualization - {doc_name}</h2>
        <div class="legend">
            <div class="legend-item">
                <span class="color-box" style="background: rgb(0, 255, 0);"></span>
                <span>Low NLL (Green)</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgb(255, 255, 0);"></span>
                <span>Medium NLL (Yellow)</span>
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: rgb(255, 0, 0);"></span>
                <span>High NLL (Red)</span>
            </div>
        </div>
    """
    # Tokenize and visualize this line
    token_ids = tokenizer.encode(text_content, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens_str = [tokenizer.convert_tokens_to_string([token]) for token in tokens]

    if len(token_ids) != len(all_nlls):
        print(f"Warning: length mismatch in {doc_name}")
        print(f"token_ids length: {len(token_ids)}, nlls length: {len(all_nlls)}")
        print(f"tokens_str: {tokens_str}")
        return
    
    # Normalize NLL values using global min/max
    if global_nll_max == global_nll_min:
        normalized_nlls = [0.5] * len(all_nlls)
    else:
        normalized_nlls = [(nll - global_nll_min) / (global_nll_max - global_nll_min) for nll in all_nlls]
    
    # Add sentence to HTML
    combined_html += f'<div class="sentence"><div class="sentence-number">Sentence {1}:</div>'
    
    for i, (token, nll, norm_nll) in enumerate(zip(tokens_str, all_nlls, normalized_nlls)):
        display_token = token
        if display_token.strip():
            # Check if position should be masked
            if masked_positions is not None and i in masked_positions:
                # Don't color masked positions
                combined_html += f'<span class="token" title="NLL: {nll:.4f} (masked)">{display_token}</span>'
            else:
                # Color non-masked positions
                color = nll_to_rgb(norm_nll)
                combined_html += f'<span class="token" style="background-color: {color};" title="NLL: {nll:.4f}">{display_token}</span>'
        # print(tokens[i], tokens[i].endswith('Ċ')) # debug
        if tokens[i].endswith('Ċ'):
            combined_html += '<br>'
    
    combined_html += f'</div>'
    
    combined_html += """
        </body>
        </html>
    """
        
    # Save to file
    if output_file is None:
        output_file = os.path.join(output_dir, f"{doc_name}_viz.html")
    else:
        output_file = os.path.join(output_dir, output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_html)
    print(f"Visualization saved to {output_file}")


# %%
def exp_poem():
    text_dir = 'text_data/poem_Human'
    nll_dir = 'nll_data/poem_Human'
    output_dir = 'poem_Human_output'
    for i in range(0,10):
        run_poem_viz_doc(f'{i}', text_dir, nll_dir, tokenizer, output_dir=output_dir) 

    text_dir = 'text_data/poem_ChatGPT'
    nll_dir = 'nll_data/poem_ChatGPT'
    output_dir = 'poem_ChatGPT_output'
    for i in range(0,10):
        run_poem_viz_doc(f'{i}', text_dir, nll_dir, tokenizer, output_dir=output_dir)

    text_dir = 'text_data/poem_Tulu2'
    nll_dir = 'nll_data/poem_Tulu2'
    output_dir = 'poem_Tulu2_output'
    for i in range(0,10):
        run_poem_viz_doc(f'{i}', text_dir, nll_dir, tokenizer, output_dir=output_dir)


# %%
def exp_poem_mask():
    # Experiment with 1.txt for Human, ChatGPT, Tulu2
    # masking the first line in the final viz
    # and using the NLLs from the unmasked tokens to compute the global min and max
    
    # Load data for all three sources
    text_dir_human = 'text_data/poem_Human'
    nll_dir_human = 'nll_data/poem_Human'
    text_dir_chatgpt = 'text_data/poem_ChatGPT'
    nll_dir_chatgpt = 'nll_data/poem_ChatGPT'
    text_dir_tulu2 = 'text_data/poem_Tulu2'
    nll_dir_tulu2 = 'nll_data/poem_Tulu2'
    
    # Read NLL data for computing global range
    with open(os.path.join(nll_dir_human, '1.txt'), 'r') as f:
        nll_human = np.array(list(map(float, f.read().strip().split())))
    
    with open(os.path.join(nll_dir_chatgpt, '1.txt'), 'r') as f:
        nll_chatgpt = np.array(list(map(float, f.read().strip().split())))
    
    with open(os.path.join(nll_dir_tulu2, '1.txt'), 'r') as f:
        nll_tulu2 = np.array(list(map(float, f.read().strip().split())))
    
    # Define masked positions (first line tokens)
    masked_positions = np.array(list(range(0, 10)), dtype=np.int32)
    
    # Compute global min/max excluding masked positions
    global_nll_min = np.min(np.concatenate([
        nll_human[~masked_positions], 
        nll_chatgpt[~masked_positions], 
        nll_tulu2[~masked_positions]
    ]))
    global_nll_max = np.max(np.concatenate([
        nll_human[~masked_positions], 
        nll_chatgpt[~masked_positions], 
        nll_tulu2[~masked_positions]
    ]))
    
    print(f"Global NLL range (excluding masked): {global_nll_min:.4f} to {global_nll_max:.4f}")
    
    # Create output directories
    output_dir = 'poem_output_masked'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations with masking
    run_poem_viz_doc('1', text_dir_human, nll_dir_human, tokenizer, 
                     output_dir, '1_Human_viz.html', global_nll_min, global_nll_max, masked_positions)
    run_poem_viz_doc('1', text_dir_chatgpt, nll_dir_chatgpt, tokenizer, 
                     output_dir, '1_ChatGPT_viz.html', global_nll_min, global_nll_max, masked_positions)
    run_poem_viz_doc('1', text_dir_tulu2, nll_dir_tulu2, tokenizer, 
                     output_dir, '1_Tulu2_viz.html', global_nll_min, global_nll_max, masked_positions)


# %%
# Test the visualization
if __name__ == "__main__":
    # Load tokenizer
    model_path = '/Users/xy/models/llama3-8b-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Experiment with Poem
    # exp_poem()
    
    # Experiment with Poem with masking
    exp_poem_mask()
