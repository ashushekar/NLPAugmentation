"""
Using transformers to augment any given company name. We will use transformers library for this purpose.
"""

import random

# 1. Random insertion
from transformers import pipeline
import pandas as pd

pd.set_option('display.max_columns', None)

# remove [SEP] and [CLS] tokens
def remove_special_tokens(text):
    return text.replace('[CLS]', '').replace('[SEP]', '').strip()

unmasker = pipeline('fill-mask', model='bert-large-uncased')

company_name = ["CEDAR HILL INDEPENDENT SCHOOL DISTRICT", "INMOBILIARIA OSNA SA DE CV",
                "UNITIMBER AB", "AXON PUBLIC SAFETY INC", "BANK OF AMERICA CORPORATION"]
df = pd.DataFrame({'company_name': company_name})


def augment(company_name):
    orig_text_list = company_name.split()
    len_input = len(orig_text_list)
    if len_input == 1:
        return [company_name]
    all_aug_text = []
    for rand_idx in range(1, len_input):
        # for insertion
        new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
        new_mask_sent = ' '.join(new_text_list)

        augmented_text_list = unmasker(new_mask_sent)
        print(augmented_text_list)
        augmented_text = [d['sequence'].upper() for d in augmented_text_list]
        all_aug_text.append(augmented_text)

        # for replacement
        new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx+1:]
        new_mask_sent = ' '.join(new_text_list)
        augmented_text_list = unmasker(new_mask_sent)
        augmented_text = [d['sequence'].upper() for d in augmented_text_list if d['token_str'] not in
                          ['.', '|', '!', '?']]
        all_aug_text.append(augmented_text)

    # drop duplicates and flatten the list
    all_aug_text = list(set([item for sublist in all_aug_text for item in sublist]))

    return all_aug_text


df['augmented_company_name'] = df['company_name'].apply(augment)
# Transpose the dataframe
df = df.explode('augmented_company_name')
df = df.reset_index(drop=True)
print(df)