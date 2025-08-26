# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# %%
# Load tokenizer
model_path = '/Users/xy/models/llama3-8b-base'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


# %%
text_dir = 'text_data/wsj_cleaned/'
nll_dir = 'nll_data/wsj_llama3-8b-base'

input_filename = 'wsj_0001.txt'
input_text_file = os.path.join(text_dir, input_filename)
input_nll_file = os.path.join(nll_dir, input_filename)

assert os.path.exists(input_text_file)
assert os.path.exists(input_nll_file)


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
def run_viz(texts, nlls, tokenizer, output_file=None):
    """
    texts: str
    nlls: list[float], or numpy array
    tokenizer: tokenizer object
    output_file: str, optional output HTML file path
    """
    if isinstance(nlls, np.ndarray):
        nlls = nlls.tolist()
    
    # tokenize texts
    token_ids = tokenizer.encode(texts, add_special_tokens=False)
    if len(token_ids) != len(nlls):
        print(f"Error: length of token_ids ({len(token_ids)}) and nlls ({len(nlls)}) do not match")
        return
    
    # decode tokens back to text for visualization
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # normalize NLL values for color mapping (0-1 range)
    nll_min, nll_max = min(nlls), max(nlls)
    if nll_max == nll_min:
        normalized_nlls = [0.5] * len(nlls)  # all same color if all values are identical
    else:
        normalized_nlls = [(nll - nll_min) / (nll_max - nll_min) for nll in nlls]
    
    # generate HTML with colored background
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>NLL Visualization</title>
        <style>
            body { font-family: 'Courier New', monospace; font-size: 16px; line-height: 1.6; margin: 20px; }
            .token { padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; }
            .legend { margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }
            .legend-item { display: inline-block; margin-right: 20px; }
            .color-box { display: inline-block; width: 20px; height: 20px; margin-right: 5px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h2>NLL Visualization</h2>
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
        <div style="background: white; padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
    """
    
    # add colored tokens
    for i, (token, nll, norm_nll) in enumerate(zip(tokens, nlls, normalized_nlls)):
        color = nll_to_rgb(norm_nll)
        # clean up token display (remove special characters)
        display_token = token.replace('▁', ' ').replace('</s>', '').replace('<s>', '').replace('Ġ', '')
        if display_token.strip():  # only show non-empty tokens
            html_content += f'<span class="token" style="background-color: {color};" title="NLL: {nll:.4f}">{display_token}</span>'
    
    html_content += """
        </div>
        <div style="margin-top: 20px; font-size: 12px; color: #666;">
            <p>NLL Range: {:.4f} to {:.4f}</p>
            <p>Hover over tokens to see exact NLL values</p>
        </div>
    </body>
    </html>
    """.format(nll_min, nll_max)
    
    # save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Visualization saved to {output_file}")
    
    # also print to console for immediate viewing
    print("Generated HTML visualization:")
    print(html_content)
    
    return html_content


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
def run_chuci_viz_doc(doc_name, text_dir, nll_dir, tokenizer, global_nll_min=None, global_nll_max=None):
    """
    Note: There are multiple lines in chuci text file, while only one line in nll file
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
    tokens_zh = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    # tokens_zh = [tokenizer.decode([token_id]) for token_id in token_ids] # alternative

    if len(token_ids) != len(all_nlls):
        print(f"Warning: length mismatch in {doc_name}")
        print(f"token_ids length: {len(token_ids)}, nlls length: {len(all_nlls)}")
        print(f"tokens_zh: {tokens_zh}")
        return
    
    # Normalize NLL values using global min/max
    if global_nll_max == global_nll_min:
        normalized_nlls = [0.5] * len(all_nlls)
    else:
        normalized_nlls = [(nll - global_nll_min) / (global_nll_max - global_nll_min) for nll in all_nlls]
    
    # Add sentence to HTML
    combined_html += f'<div class="sentence"><div class="sentence-number">Sentence {1}:</div>'
    
    for token, nll, norm_nll in zip(tokens_zh, all_nlls, normalized_nlls):
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
def exp_chuci():
    text_dir = 'text_data/chuci'
    nll_dir = 'nll_data/chuci_qwen2.5-7b-base'
    tokenizer_path = '/Users/xy/models/qwen2.5-7b-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    run_chuci_viz_doc("0", text_dir, nll_dir, tokenizer)
    for i in range(1, 11):
        run_chuci_viz_doc(f"{i}", text_dir, nll_dir, tokenizer)


# %%
def test():
    texts_str = open(input_text_file, 'r').read()
    nlls_str = open(input_nll_file, 'r').read()
    # print(texts_str)
    # print(nlls_str)
    texts0 = texts_str.split('\n')[0]
    nlls0 = nlls_str.split('\n')[0]
    nlls0 = list(map(float, nlls0.split()))
    print('texts0: ', texts0)
    print('nlls0: ', nlls0)

    result = run_viz(texts0, nlls0, tokenizer, "nll_visualization.html")

# %%
# Test the visualization
if __name__ == "__main__":
    # Test with the first sentence
    # test()

    # Experiment with WSJ
    # exp_wsj()

    # Experiment with Chuci
    exp_chuci()
    