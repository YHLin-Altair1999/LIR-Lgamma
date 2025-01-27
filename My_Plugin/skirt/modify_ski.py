import numpy as np
import astropy.units as u
import astropy.constants as c

def modify_skifile(input_file, output_file, width):

    # Define a dictionary of words to search for and their replacements
    replacements = {
        'minX="-3e4 pc"': f'minX="{(-width[0]/2).to("pc").value:.5e} pc"',
        'maxX="3e4 pc"' : f'maxX="{(+width[0]/2).to("pc").value:.5e} pc"',
        'minY="-3e4 pc"': f'minY="{(-width[1]/2).to("pc").value:.5e} pc"',
        'maxY="3e4 pc"' : f'maxY="{(+width[1]/2).to("pc").value:.5e} pc"',
        'minZ="-3e4 pc"': f'minZ="{(-width[2]/2).to("pc").value:.5e} pc"',
        'maxZ="3e4 pc"' : f'maxZ="{(+width[2]/2).to("pc").value:.5e} pc"',
        'fieldOfViewX="3e4 pc"': f'fieldOfViewX="{width[0].to("pc").value:.5e} pc"',
        'fieldOfViewY="3e4 pc"': f'fieldOfViewY="{width[0].to("pc").value:.5e} pc"'
        }

    # Read the file
    with open(input_file, "r") as file:
        content = file.read()

    # Replace each word in the dictionary
    for old_word, new_word in replacements.items():
        content = content.replace(old_word, new_word)

    # Write the modified content to the output file
    with open(output_file, "w") as file:
        file.write(content)

    print(f"Replacements complete. Modified text saved to '{output_file}'")
    return None
