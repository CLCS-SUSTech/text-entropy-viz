# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# %%
def nll_to_rgb(value):
    """Convert normalized NLL value (0-1) to RGB color string"""
    # Use a diverging colormap: green (low NLL) to yellow (medium) to red (high NLL)
    # 0 = green, 0.5 = yellow, 1 = red
    if value <= 0.5:
        # Green to yellow
        intensity = 1 - 2 * value
        return f"rgb({int(255 * (1 - intensity))}, 255, 0)"
    else:
        # Yellow to red
        intensity = 2 * (value - 0.5)
        return f"rgb(255, {int(255 * (1 - intensity))}, 0)"


# %%
def run_poem_viz_doc(doc_name, text_dir, nll_dir, tokenizer, output_dir='.', global_nll_min=None, global_nll_max=None):
    """
    Note: There are multiple lines in poem text file, while only one line in nll file
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
        color = nll_to_rgb(norm_nll)
        display_token = token
        if display_token.strip():
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
    output_file = os.path.join(output_dir, f"{doc_name}_viz.html")
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
# Test the visualization
if __name__ == "__main__":
    # Load tokenizer
    model_path = '/Users/xy/models/llama3-8b-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Experiment with Poem
    exp_poem()
