# %%
import os
from transformers import pipeline, set_seed

# %%
# Local model path 
model_path = '/Users/xy/models/gpt2-xl'
assert os.path.exists(model_path), f'Model path {model_path} does not exist'


def exp_gpt2():
    # %%
    generator = pipeline('text-generation', model=model_path)
    set_seed(42)

    # %%
    # WSJ sentence as a prompt
    prompt = 'Sheraton Corp. and Pan American World Airways announced that'

    # The original complete sentence is:
    # Sheraton Corp. and Pan American World Airways announced that they and two Soviet partners will construct two "world-class" hotels within a mile of Red Square in Moscow.

    outputs = generator(prompt, max_new_tokens=30, num_return_sequences=5)

    for i, item in enumerate(outputs):
        print(f'{i}: {item["generated_text"]}')

# %%
# 0: Sheraton Corp. and Pan American World Airways announced that the $2.3 billion merger would create a new company with $8.4 billion in revenue and a market value of $9.2 billion
# 1: Sheraton Corp. and Pan American World Airways announced that they were joining forces, forming Pan Am AVIATION PLC, a U.S. subsidiary that will operate Pan Am's international business as well
# 2: Sheraton Corp. and Pan American World Airways announced that they had reached an agreement for the merger. The new company, known as Pan American World Airways (PAA), would be a member of the United
# 3: Sheraton Corp. and Pan American World Airways announced that they have agreed to merge. The merger will create a new company, the combined Marriott/Hyatt Corp. The deal is expected to close
# 4: Sheraton Corp. and Pan American World Airways announced that they would be closing flights to Miami and Fort Lauderdale.

# "The decision to close service to Miami and Fort Lauderdale is not easy, but it


# %%
# Larger model

def exp_llama3():
    llama3_path = '/Users/xy/models/llama3-8b-base'
    generator = pipeline('text-generation', model=llama3_path)
    set_seed(42)

    # %%
    prompt = 'Sheraton Corp. and Pan American World Airways announced that'
    outputs = generator(prompt, max_new_tokens=30, num_return_sequences=5, temperature=1.0, top_p=0.9, top_k=200)

    for i, item in enumerate(outputs):
        print(f'{i}: {item["generated_text"]}')

# 0: Sheraton Corp. and Pan American World Airways announced that they had signed a letter of intent to form a joint venture to manage and operate a chain of hotels in the United States and overseas. The new company
# 1: Sheraton Corp. and Pan American World Airways announced that the hotel chain has joined the Diners Club Travelers' Club program. The program, established by Diners Club International, offers frequent travelers the ability
# 2: Sheraton Corp. and Pan American World Airways announced that they have signed an agreement for the Sheraton to manage the Pan Am Building at 200 Park Avenue, New York, N.Y. The agreement,
# 3: Sheraton Corp. and Pan American World Airways announced that they have signed an agreement for the Sheraton to operate the new 500-room Sheraton Pan Am Hotel in New York. The hotel will be located
# 4: Sheraton Corp. and Pan American World Airways announced that they have reached agreement for Sheraton to operate a hotel at the new Pan American terminal at John F. Kennedy International Airport. The hotel will have

# %%
if __name__ == "__main__":
    # exp_gpt2()
    exp_llama3()
