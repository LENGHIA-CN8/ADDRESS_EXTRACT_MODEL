import re
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm_notebook
import json
import random

SEED_VALUE = 42
random.seed(SEED_VALUE)

def filter_address(text):
    # Use regex to extract text after "Địa chỉ:"
    match = re.search(r'Địa chỉ:\s*(.*)', text)

    if match:
        address_info = match.group(1)
    else:
        return ''
    return address_info

def create_df(list_add):
    address_df = pd.DataFrame()
    for add in tqdm(list_add):
        try:
            std_address = []
            for std_a in add['administrative_units']:
                if std_a.get('level', '') in ['Đường/Phố', 'Đường','Phường', 'Xã', 'Huyện', 'Thành phố', 'Tỉnh']:
                    std_address.append(std_a.get('org_prefix', '') + ' ' + std_a.get('name', ''))
            std_address = ', '.join(std_address)
            add_row = pd.DataFrame({'input_address': [filter_address(add['address'])], 'filter_address': [std_address]})
            address_df = pd.concat([address_df, add_row], ignore_index=True)
            
            list_std_address = [a.strip() for a in std_address.split(',')]
            if len(list_std_address) <= 1:
                continue
            ## coin choose
            coin_result = random.choice([0, 1])
            # Check the result
            if coin_result == 1:
                continue
                
            idx = random.randint(1, len(list_std_address) - 1)
            input_address = list_std_address[idx:]
            if len(input_address) <= 0:
                continue
            add_row = pd.DataFrame({'input_address': [', '.join(input_address).strip(', ')], 'filter_address': [', '.join(input_address).strip(', ')]})
            address_df = pd.concat([address_df, add_row], ignore_index=True)
        except Exception as e:
            print(e)
            break
    return address_df

if __name__ == "__main__":
    file_path = './data/cty_address_au.json'
    list_add = []
    try:
        with open(file_path) as f:
            for line in f:
                d = json.loads(line)
                list_add.append(d)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    data_df = create_df(list_add)
    data_df.to_csv('./data/address_raw_data.csv', index=False)