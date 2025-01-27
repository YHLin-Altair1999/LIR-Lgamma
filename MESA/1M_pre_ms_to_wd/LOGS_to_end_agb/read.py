import pandas as pd

file_path = './profile1.data'
header_line_number = 5  # Adjust to the correct line number (0-indexed)
with open(file_path, 'r') as f:
    lines = f.readlines()
    column_names = lines[header_line_number].strip().split()  # Adjust separator if needed

# Step 2: Read the FWF file using the extracted column names
df = pd.read_fwf(file_path, skiprows=header_line_number + 1, names=column_names)

print(df)




