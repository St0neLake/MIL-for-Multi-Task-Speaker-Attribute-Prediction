#!/bin/bash

# Get the directory of this script and add the "hein-daily" directory to it
hein_daily_raw_data_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/hein-daily"

# Echo hein_daily_raw_data_dir
echo "hein_daily_raw_data_dir: ${hein_daily_raw_data_dir}"

# Get the directory of this script
congress_terms_raw_data_address="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/congress-terms.csv"

# Echo congress_terms_raw_data_address
echo "congress_terms_raw_data_address: ${congress_terms_raw_data_address}"

preprocessed_data_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/preprocessed_data"

# Echo preprocessed_data_dir
echo "preprocessed_data_dir: ${preprocessed_data_dir}"

# Get the grand parent directory of this script data/political_data.csv
final_data_address="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd ../.. && pwd )/data/political_data.csv"

# Echo final_data_address
echo "final_data_address: ${final_data_address}"

# Create a list of years to process. By reading all of the .txt files in the hein_daily_raw_data_dir directory which start with "speeches_{congress}.txt" and extracting the congress number from the filename.
years=$(ls ${hein_daily_raw_data_dir} | grep "speeches_[0-9]*.txt" | sed -e 's/speeches_//' -e 's/.txt//')

python3 preprocess.py \
  --hein_daily_raw_data_dir ${hein_daily_raw_data_dir} \
  --congress_terms_raw_data_address ${congress_terms_raw_data_address} \
  --hein_daily_preprocessed_data_dir ${preprocessed_data_dir} \
  --final_data_address ${final_data_address} \
  --years ${years} 