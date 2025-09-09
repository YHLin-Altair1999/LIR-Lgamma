import numpy as np

N = 40
inc_max = np.pi/3
inc = inc_max*np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
az  = np.linspace(0, 2*np.pi, N, endpoint=False)

# convert to degrees and prepare formatted instrument lines
az_deg = np.degrees(az)
inc_deg = np.degrees(inc)

lines = []
for i in range(N):
    # round and ensure azimuth is in [0,360)
    az_i = int(round(az_deg[i])) % 360
    inc_i = int(round(inc_deg[i])) + 90
    lines.append(
        f'<FrameInstrument instrumentName="p{az_i:03d}" distance="1 Mpc" '
        f'inclination="{inc_i:03d} deg" azimuth="{az_i:03d} deg" roll="0 deg" '
        f'fieldOfViewX="4e4 pc" numPixelsX="800" centerX="0 pc" '
        f'fieldOfViewY="4e4 pc" numPixelsY="800" centerY="0 pc" '
        f'recordComponents="false" numScatteringLevels="0" '
        f'recordPolarization="false" recordStatistics="false"/>'
    )


def insert_instruments(template_path='mov_template.ski', out_path='mov.ski'):
    """Read template, replace contents between the instruments tags, and write output."""
    start_tag = '<instruments type="Instrument">'
    end_tag = '</instruments>'

    with open(template_path, 'r') as fh:
        txt = fh.read()

    start_idx = txt.find(start_tag)
    if start_idx == -1:
        raise RuntimeError(f'start tag not found: {start_tag}')
    end_idx = txt.find(end_tag, start_idx)
    if end_idx == -1:
        raise RuntimeError(f'end tag not found: {end_tag}')

    # detect indentation of the opening tag and reuse it for inserted lines
    line_start = txt.rfind('\n', 0, start_idx) + 1
    indent = txt[line_start:start_idx]
    if not indent:
        indent = '    '

    # add one extra indent level so the inserted lines align inside the instruments block
    insert_block = '\n'.join(indent + '    ' + l for l in lines)

    new_txt = txt[: start_idx + len(start_tag)] + '\n' + insert_block + '\n' + txt[end_idx:]

    with open(out_path, 'w') as fh:
        fh.write(new_txt)

    print(f'Wrote {out_path} with {len(lines)} FrameInstrument entries.')


if __name__ == '__main__':
    insert_instruments()

