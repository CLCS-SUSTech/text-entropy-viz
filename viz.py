# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
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
def run_viz(text_file, nll_file, tokenizer, output_file=None, global_nll_min=None, global_nll_max=None, masked_positions=None):
    """
    Visualize NLL values for text from files (single line)
    
    Args:
        text_file: path to text file (contains single line)
        nll_file: path to NLL file (contains single line)
        tokenizer: tokenizer object
        output_file: str, optional output HTML file path
        global_nll_min: float, optional global minimum NLL value for normalization
        global_nll_max: float, optional global maximum NLL value for normalization
        masked_positions: list of int, optional positions to mask in the text, which will not be colored
    """
    # Read text and NLL data
    with open(text_file, 'r', encoding='utf-8') as f:
        text_content = f.read().strip()
    
    with open(nll_file, 'r', encoding='utf-8') as f:
        nll_content = f.read().strip()
    
    # Process single line
    nlls = list(map(float, nll_content.split()))
    
    # Compute global min/max if not provided
    if global_nll_min is None:
        global_nll_min = min(nlls)
    if global_nll_max is None:
        global_nll_max = max(nlls)
    
    print(f"Global NLL range: {global_nll_min:.4f} to {global_nll_max:.4f}")
    
    # Tokenize and visualize
    token_ids = tokenizer.encode(text_content, add_special_tokens=False)
    if len(token_ids) != len(nlls):
        print(f"Error: length mismatch")
        print(f"token_ids length: {len(token_ids)}, nlls length: {len(nlls)}")
        return
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Normalize NLL values using global min/max
    if global_nll_max == global_nll_min:
        normalized_nlls = [0.5] * len(nlls)
    else:
        normalized_nlls = [(nll - global_nll_min) / (global_nll_max - global_nll_min) for nll in nlls]
    
    # Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>NLL Visualization</title>
        <style>
            body {{ font-family: 'Courier New', monospace; font-size: 16px; line-height: 1.6; margin: 20px; }}
            .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; }}
            .legend {{ margin: 20px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
            .legend-item {{ display: inline-block; margin-right: 20px; }}
            .color-box {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; border: 1px solid #ccc; }}
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
    
    # Add colored tokens
    for i, (token, nll, norm_nll) in enumerate(zip(tokens, nlls, normalized_nlls)):
        display_token = token.replace('▁', ' ').replace('</s>', '').replace('<s>', '').replace('Ġ', '')
        if display_token.strip():
            # Check if position should be masked
            if masked_positions is not None and i in masked_positions:
                # Don't color masked positions
                html_content += f'<span class="token" title="NLL: {nll:.4f} (masked)">{display_token}</span>'
            else:
                # Color non-masked positions
                color = nll_to_rgb(norm_nll)
                html_content += f'<span class="token" style="background-color: {color};" title="NLL: {nll:.4f}">{display_token}</span>'
    
    html_content += """
        </div>
        <div style="margin-top: 20px; font-size: 12px; color: #666;">
            <p>NLL Range: {:.4f} to {:.4f}</p>
            <p>Hover over tokens to see exact NLL values</p>
        </div>
    </body>
    </html>
    """.format(global_nll_min, global_nll_max)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Visualization saved to {output_file}")
    
    return html_content


# %%
def run_viz_doc(text_file, nll_file, tokenizer, output_file=None, global_nll_min=None, global_nll_max=None, masked_positions=None):
    """
    Visualize NLL values for text from files
    
    Args:
        text_file: path to text file
        nll_file: path to NLL file
        tokenizer: tokenizer object
        output_file: str, optional output HTML file path
        global_nll_min: float, optional global minimum NLL value for normalization
        global_nll_max: float, optional global maximum NLL value for normalization
        masked_positions: list of int, optional positions to mask in the text, which will not be colored
    """
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
    
    # Compute global min/max if not provided
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
        <title>NLL Visualization</title>
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
    """
        
    for i, (text_line, nll_line) in enumerate(zip(text_lines, nll_lines)):
        if not text_line.strip() or not nll_line.strip():
            continue
            
        nlls = list(map(float, nll_line.split()))
        
        # Tokenize and visualize this line
        token_ids = tokenizer.encode(text_line, add_special_tokens=False)
        if len(token_ids) != len(nlls):
            print(f"Warning: length mismatch in line {i+1}")
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
        
        for j, (token, nll, norm_nll) in enumerate(zip(tokens, nlls, normalized_nlls)):
            display_token = token.replace('▁', ' ').replace('</s>', '').replace('<s>', '').replace('Ġ', '')
            if display_token.strip():
                # Check if position should be masked
                if masked_positions is not None and j in masked_positions:
                    # Don't color masked positions
                    combined_html += f'<span class="token" title="NLL: {nll:.4f} (masked)">{display_token}</span>'
                else:
                    # Color non-masked positions
                    color = nll_to_rgb(norm_nll)
                    combined_html += f'<span class="token" style="background-color: {color};" title="NLL: {nll:.4f}">{display_token}</span>'
        
        combined_html += f'</div>'
    
    combined_html += """
        </body>
        </html>
    """
        
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_html)
        print(f"Visualization saved to {output_file}")
    
    return combined_html


# %%
# Load tokenizer
model_path = '/Users/xy/models/llama3-8b-base'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


# %%
# Test the visualization
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize NLL values for text')
    parser.add_argument('--text', required=True, help='Path to text file')
    parser.add_argument('--nll', required=True, help='Path to NLL file')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.text):
        print(f"Error: Text file not found: {args.text}")
        exit(1)
    
    if not os.path.exists(args.nll):
        print(f"Error: NLL file not found: {args.nll}")
        exit(1)
    
    # Run visualization
    run_viz(args.text, args.nll, tokenizer, args.output)
    