import os
import re
from glob import glob

#folders = list(glob('./LOGS_*/history.data'))
folders = [
    './LOGS_start/',
    './LOGS_to_end_core_h_burn/',
    './LOGS_to_start_he_core_flash/',
    './LOGS_to_end_core_he_burn/',
    './LOGS_to_end_agb/',
    './LOGS_to_wd/',
    ]

output_folder = './LOGS_combined/'

# Deal with history
f0 = open(os.path.join(folders[0], 'history.data')).readlines()
for folder in folders[1:]:
    f0.extend(open(os.path.join(folder, 'history.data')).readlines()[6:])

output_file = os.path.join(output_folder, 'history.data')
with open(output_file, 'w') as outfile:
    outfile.writelines(f0)

# Deal with profiles

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]

profile_id = 1
for folder in folders:
    profiles = glob(folder+'profile*.data')
    profiles = sorted(profiles, key=natural_sort_key)
    for profile in profiles:
        fname = os.path.join(output_folder, f'profile{profile_id}.data')
        os.system(f'cp {profile} {fname}')
        profile_id += 1

