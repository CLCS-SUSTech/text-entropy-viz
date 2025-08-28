# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from viz import nll_to_rgb, run_viz

# %%
def run_wsj_viz_doc(doc_name, text_dir, nll_dir, tokenizer, global_nll_min=None, global_nll_max=None):
    """
    Visualize a single document with all its sentences
    
    Args:
        doc_name: document base name (e.g., "wsj_1126")
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
    
    # Process each line
    text_lines = text_content.split('\n')
    nll_lines = nll_content.split('\n')
    
    # Collect all NLL values to compute global min/max
    all_nlls = []
    for nll_line in nll_lines:
        if nll_line.strip():
            nlls = list(map(float, nll_line.split()))
            all_nlls.extend(nlls)
    
    # Compute global min/max
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
        
    for i, (text_line, nll_line) in enumerate(zip(text_lines, nll_lines)):
        if not text_line.strip() or not nll_line.strip():
            continue
            
        nlls = list(map(float, nll_line.split()))
        
        # Tokenize and visualize this line
        token_ids = tokenizer.encode(text_line, add_special_tokens=False)
        if len(token_ids) != len(nlls):
            print(f"Warning: length mismatch in {doc_name}, line {i+1}")
            print(f"token_ids length: {len(token_ids)}, nlls length: {len(nlls)}")
            continue
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Normalize NLL values using global min/max
        if global_nll_max == global_nll_min:
            normalized_nlls = [0.5] * len(nlls)
        else:
            normalized_nlls = [(nll - global_nll_min) / (global_nll_max - global_nll_min) for nll in nlls]
        
        # Add sentence to HTML
        combined_html += f'<div class="sentence"><div class="sentence-number">Sentence {i+1}:</div>'
        
        for token, nll, norm_nll in zip(tokens, nlls, normalized_nlls):
            color = nll_to_rgb(norm_nll)
            display_token = token.replace('▁', ' ').replace('</s>', '').replace('<s>', '').replace('Ġ', '')
            if display_token.strip():
                combined_html += f'<span class="token" style="background-color: {color};" title="NLL: {nll:.4f}">{display_token}</span>'
        
        combined_html += f'</div>'
    
    combined_html += """
        </body>
        </html>
    """
        
    # Save to file
    output_file = f"{doc_name}_viz.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_html)
    print(f"Visualization saved to {output_file}")


# %%
def exp_wsj():
    # run_viz_doc("wsj_1126", text_dir, nll_dir, tokenizer)
    run_wsj_viz_doc("wsj_1127", text_dir, nll_dir, tokenizer)


# %%
def exp_comparison():
    # Human text and nll
    text_human_file = 'text_data/wsj_1126_0_human.txt'
    nll_human_file = 'nll_data/wsj_1126_0_human.txt'
    nll_human_str = open(nll_human_file, 'r').read()
    nll_human = np.array(list(map(float, nll_human_str.split())))

    # Read the gpt2-generated sentence
    text_gpt2_file = 'text_data/wsj_1126_0_gpt2.txt'
    nll_gpt2_file = 'nll_data/wsj_1126_0_gpt2.txt'
    nll_gpt2_str = open(nll_gpt2_file, 'r').read()
    nll_gpt2 = np.array(list(map(float, nll_gpt2_str.split())))

    # Get global min and max for nll
    # global_nll_min = min(np.concatenate([nll_human, nll_gpt2]))
    # global_nll_max = max(np.concatenate([nll_human, nll_gpt2]))

    # Do viz respectively with masked positions
    masked_positions = np.array(list(range(0, 11)), dtype=np.int32)
    # global_nll_min = np.min(np.concatenate([nll_human[~masked_positions], nll_gpt2[~masked_positions]]))
    # global_nll_max = np.max(np.concatenate([nll_human[~masked_positions], nll_gpt2[~masked_positions]]))
    # run_viz(text_human_file, nll_human_file, tokenizer, "wsj_1126_0_human_viz.html", global_nll_min, global_nll_max, masked_positions)
    # run_viz(text_gpt2_file, nll_gpt2_file, tokenizer, "wsj_1126_0_gpt2_viz.html", global_nll_min, global_nll_max, masked_positions)

    # Read the llama3-generated sentence
    for i in range(1,5):
        text_llama3_file = f'text_data/wsj_1126_0_llama3_{i}.txt'
        nll_llama3_file = f'nll_data/wsj_1126_0_llama3_{i}.txt'
        nll_llama3_str = open(nll_llama3_file, 'r').read()
        nll_llama3 = np.array(list(map(float, nll_llama3_str.split())))
        
        global_nll_min = np.min(np.concatenate([nll_human[~masked_positions], nll_gpt2[~masked_positions], nll_llama3[~masked_positions]]))
        global_nll_max = np.max(np.concatenate([nll_human[~masked_positions], nll_gpt2[~masked_positions], nll_llama3[~masked_positions]]))
        run_viz(text_llama3_file, nll_llama3_file, tokenizer, f"wsj_1126_0_llama3_{i}_viz.html", global_nll_min, global_nll_max, masked_positions)



# %%
def test():
    text_dir = 'text_data/wsj_cleaned/'
    nll_dir = 'nll_data/wsj_llama3-8b-base'
    
    input_filename = 'wsj_0001.txt'
    input_text_file = os.path.join(text_dir, input_filename)
    input_nll_file = os.path.join(nll_dir, input_filename)
    
    texts_str = open(input_text_file, 'r').read()
    nlls_str = open(input_nll_file, 'r').read()
    # print(texts_str)
    # print(nlls_str)
    texts0 = texts_str.split('\n')[0]
    nlls0 = nlls_str.split('\n')[0]
    nlls0 = list(map(float, nlls0.split()))
    print('texts0: ', texts0)
    print('nlls0: ', nlls0)



# %%
# Test the visualization
if __name__ == "__main__":
    # Load tokenizer
    model_path = '/Users/xy/models/llama3-8b-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Set up WSJ data paths
    text_dir = 'text_data/wsj_cleaned/'
    nll_dir = 'nll_data/wsj_llama3-8b-base'
    
    # Test with the first sentence
    # test()

    # Experiment with WSJ
    # exp_wsj()

    # Comparison experiment
    exp_comparison()
