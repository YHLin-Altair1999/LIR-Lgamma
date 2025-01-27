import matplotlib.pyplot as plt
import mesaPlot as mp
import tulips
from multiprocessing import Pool
from tqdm import tqdm
import os
import argparse
from My_Plugin.mesa.plot import rhoT_plot

plt.rcParams.update(
    {
        "text.usetex": True, 
        "text.latex.preamble": r"\usepackage{amsmath}", 
        'font.family': 'STIXGeneral'
    }
    )

def plot_frame(args):
    output_index, index, plot_type = args
    try:
        m = mp.MESA()
        m.loadHistory(filename_in=EXAMPLE_DIR + 'history.data')
        m.log_fold = EXAMPLE_DIR
        
        match plot_type:
            case 'both':
                fig, axes = plt.subplots(figsize=(11,4.5), ncols=2, nrows=1) # for horizontal
                fig, axes[0] = tulips.perceived_color(m, time_ind=index, fig=fig, ax=axes[0], time_unit="kyr")
                fig, axes[1] = tulips.energy_and_mixing(
                    m, time_ind=index, 
                    cmin=-10, cmax=10, 
                    show_total_mass=True, show_mix=True, show_mix_legend=True, raxis="log_R",
                    fig=fig, ax=axes[1], time_unit="kyr")
                #fig, axes[1] = rhoT_plot(m, index, fig, axes[1])

            case 'color':
                fig, ax = plt.subplots(figsize=(5,4.5))
                fig, ax = tulips.perceived_color(m, time_ind=index, fig=fig, ax=ax, time_unit="kyr")
            case 'energy':
                fig, ax = plt.subplots(figsize=(5,4.5))
                fig, ax = tulips.energy_and_mixing(
                    m, time_ind=index, cmin=-10, cmax=10, 
                    show_total_mass=True, show_mix=True, show_mix_legend=True,
                    fig=fig, ax=ax, time_unit="kyr")
        
        folder = os.path.basename(os.path.dirname(os.path.dirname(EXAMPLE_DIR)))
        filename = f'images/{folder}/{output_index:04d}.png'
        
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close('all')
        return True
    except Exception as e:
        print(f"Error processing index {index}: {str(e)}")
        return False

if __name__ == "__main__":
    EXAMPLE_DIR = "./1M_pre_ms_to_wd/LOGS_combined/"
    #EXAMPLE_DIR = "./12M_pre_ms_to_core_collapse/LOGS/"
    m = mp.MESA()
    m.loadHistory(filename_in=EXAMPLE_DIR + 'history.data')

    indices = list(range(len(m.hist.star_age)))[::6]
    input_args = list(enumerate(indices))
    plot_type = 'both'
    input_args = [list(input_arg) for input_arg in input_args]
    [input_arg.append(plot_type) for input_arg in input_args]

    # Create output directory
    folder = os.path.basename(os.path.dirname(os.path.dirname(EXAMPLE_DIR)))
    os.makedirs(f'images/{folder}', exist_ok=True)

    parser = argparse.ArgumentParser(description="Run specific functions based on arguments.")
    parser.add_argument("parameter", nargs="?", default=None, help="Specify 'para' to run funcB.")
    args = parser.parse_args()
    
    if args.parameter == "para":
        with Pool() as pool:
            list(tqdm(pool.imap(plot_frame, input_args), total=len(indices)))
    else:
        plot_frame(input_args[-1])
        #for arg in tqdm(input_args):
        #    plot_frame(arg)
