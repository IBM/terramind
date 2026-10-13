
import torch
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, LinearSegmentedColormap

# Plotting utils
# 'No Data', 'Water', 'Trees', 'Flooded vegetation', 'Crops', 'Built area', 'Bare ground', 'Snow/ice', 'Clouds', 'Rangeland'
COLORBLIND_HEX = ["#000000", "#3171AD", "#469C76", '#83CA70', "#EAE159", "#C07CB8", "#C19368", "#6FB2E4", "#F1F1F1",
                  "#C66526"]
COLORBLIND_RGB = [hex2color(hex) for hex in COLORBLIND_HEX]
lulc_cmap = LinearSegmentedColormap.from_list('lulc', COLORBLIND_RGB, N=10)


def s2_to_rgb(data, smooth_quantiles=True, gamma=0.7):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        data = data[0]

    # Select
    if data.shape[0] > 13:
        # assuming channel last
        rgb = data[:, :, [3, 2, 1]]
    else:
        # assuming channel first
        rgb = data[[3, 2, 1]].transpose((1, 2, 0))

    if smooth_quantiles:
        min_value, q99_value = np.quantile(rgb, q=[0., 0.99])
        min_value = min_value.clip(0, 1000) # Clip scaling
        q99_value = q99_value.clip(2000, 20000)
        rgb = (rgb - min_value) / (q99_value - min_value + 1e-6)
    else:
        rgb = rgb / 2000

    rgb = rgb.clip(0, 1)
    if gamma is not None:
        rgb = np.power(rgb, gamma)

    # to uint8
    rgb = (rgb * 255).round().astype(np.uint8)

    return rgb


def s1_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        data = data[0]

    vv = data[0]
    vh = data[1]
    r = (vv + 30) / 40  # scale -30 to +10
    g = (vh + 40) / 40  # scale -40 to +0
    b = vv / vh.clip(-40, -1) / 1.5  # VV / VH

    rgb = np.dstack([r, g, b])
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def s1_to_power(data):
    # Convert dB to power
    data = 10 ** (data / 10)
    return data * 10000


def s1_power_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        data = data[0]

    vv = data[0]
    vh = data[1]
    r = vv / 500
    g = vh / 2200
    b = vv / vh / 2

    rgb = np.dstack([r, g, b])
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def dem_to_rgb(data, cmap='BrBG_r', buffer=5):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        data = data[0]

    # Add 10m buffer to highlight flat areas
    data_min, data_max = data.min(), data.max()
    data_min -= buffer
    data_max += buffer
    data = (data - data_min) / (data_max - data_min + 1e-6)

    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def ndvi_to_rgb(data, cmap='RdYlGn'):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        data = data[0]

    # Scale NDVI to 0-1
    data = (data + 1) / 2

    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def lulc_to_rgb(data, cmap=lulc_cmap, num_classes=10):
    while len(data.shape) > 2:
        if data.shape[0] == num_classes:
            data = data.argmax(axis=0)  # First dim are class logits
        else:
            # Remove batch dim
            data = data[0]

    rgb = cmap(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def coords_to_text(data):
    if isinstance(data, torch.Tensor):
        data = data.clone().cpu().numpy()
    if len(data.shape) > 1:
        # Remove batch dim etc.
        data = data[0]
    if data.shape[0] > 2:
        # Not coords
        return str(data)
    else:

        return f'lon={data[0]:.2f}, lat={data[1]:.2f}'


def plot_s2(data, ax=None, smooth_quantiles=True, gamma=0.7, *args, **kwargs):
    rgb = s2_to_rgb(data, smooth_quantiles=smooth_quantiles, gamma=gamma)

    if ax is None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb)
        ax.axis('off')


def plot_s1(data, ax=None, power=False, *args, **kwargs):
    if power:
        data = s1_to_power(data)
        rgb = s1_power_to_rgb(data)
    else:
        rgb = s1_to_rgb(data)

    if ax is None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb)
        ax.axis('off')


def plot_dem(data, ax=None, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        data = data[0]

    # Add 10m buffer to highlight flat areas
    data_min, data_max = data.min(), data.max()
    data_min -= 5
    data_max += 5
    data = (data - data_min) / (data_max - data_min + 1e-6)

    data = (data * 255).round().clip(0, 255).astype(np.uint8)

    if ax is None:
        plt.imshow(data, vmin=0, vmax=255, cmap='BrBG_r')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(data, vmin=0, vmax=255, cmap='BrBG_r')
        ax.axis('off')


def plot_lulc(data, ax=None, num_classes=10, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        if data.shape[0] == num_classes:
            data = data.argmax(axis=0)  # First dim are class logits
        else:
            # Remove batch dim
            data = data[0]

    if ax is None:
        plt.imshow(data, vmin=0, vmax=num_classes-1, cmap=lulc_cmap, interpolation='nearest')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(data, vmin=0, vmax=num_classes-1, cmap=lulc_cmap, interpolation='nearest')
        ax.axis('off')


def plot_ndvi(data, ax=None, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        data = data[0]

    if ax is None:
        plt.imshow(data, vmin=-1, vmax=+1, cmap='RdYlGn')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(data, vmin=-1, vmax=+1, cmap='RdYlGn')
        ax.axis('off')


def wrap_text(text, ax, font_size):
    # Get the width of the axis in pixels
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height

    # Calculate the number of characters per line
    char_width = font_size * 0.6  # Approximate width of a character
    max_chars_per_line = int(width / char_width * 0.75)
    max_lines = int(height / font_size * 0.5)

    # Wrap the text
    wrapped_text = textwrap.wrap(text, width=max_chars_per_line)

    if len(wrapped_text) > max_lines:
        wrapped_text = wrapped_text[:max_lines]
        wrapped_text[-1] += '...'

    return '\n'.join(wrapped_text)


def plot_text(data, ax=None, *args, **kwargs):
    if isinstance(data, str):
        text = data
    elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        # assuming coordinates
        text = coords_to_text(data)
    else:
        raise ValueError()

    font_size = 14 if len(text) > 150 else 20

    if ax is None:
        fig, ax = plt.subplots()
        wrapped_text = wrap_text(text, ax, font_size)
        ax.text(0.5, 0.5, wrapped_text, fontsize=font_size, ha='center', va='center', wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    else:
        wrapped_text = wrap_text(text, ax, font_size)
        ax.text(0.5, 0.5, wrapped_text, fontsize=font_size, ha='center', va='center', wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_modality(modality, data, ax=None, **kwargs):
    if 's2' in modality.lower():
        plot_s2(data, ax=ax, **kwargs)
    elif 's1' in modality.lower():
        plot_s1(data, ax=ax, **kwargs)
    elif 'dem' in modality.lower():
        plot_dem(data, ax=ax, **kwargs)
    elif 'ndvi' in modality.lower():
        plot_ndvi(data, ax=ax, **kwargs)
    elif 'lulc' in modality.lower():
        plot_lulc(data, ax=ax, **kwargs)
    elif 'coords' in modality.lower() or 'caption' in modality.lower() or 'text' in modality.lower():
        plot_text(data, ax=ax, **kwargs)
