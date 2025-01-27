from My_Plugin.skirt.convert import get_data
import h5py

fs = get_data(500)
[print(f"{attr}: {fs[0]['Header'].attrs[attr]}") for attr in fs[0]['Header'].attrs]
'''
for key, item in fs[0].items():
    print(f"Dataset: {key}")
    print(f"  Shape: {item.shape}")
    print(f"  Data type: {item.dtype}")
    print(f"  Attributes:")
    for attr in item.attrs:
        print(f"    {attr}: {item.attrs[attr]}")
    print()
'''
