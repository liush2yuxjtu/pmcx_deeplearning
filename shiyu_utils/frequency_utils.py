# @title ## 🔧 The Reusable Frequency Analyzer Toolkit (Final Class Version)
# @markdown Run this cell to define the corrected `FrequencyAnalyzer` class and see it successfully analyze both 2D and 3D data.

# ==============================================================================
# 1. INSTALL AND IMPORT LIBRARIES
# ==============================================================================
# !pip install -q nibabel scikit-image pandas

import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
import io
from scipy.ndimage import gaussian_filter, zoom
from skimage.metrics import structural_similarity as ssim

# ==============================================================================
# 2. THE FrequencyAnalyzer CLASS DEFINITION (CORRECTED)
# ==============================================================================

class FrequencyAnalyzer:
    """
    A reusable class to analyze and compare frequency content for 2D images and 3D volumes.
    """
    def __init__(self, data):
        """
        Initializes the analyzer, now with corrected 2D/3D detection.
        """
        # --- BUG FIX: Smarter dimension detection ---
        # A 3D array is only a volume if the last dimension is not a color channel.
        if data.ndim == 3 and data.shape[2] in [3, 4]:
            self.is_3d = False # It's a 2D color image
        elif data.ndim == 2:
            self.is_3d = False # It's a 2D grayscale image
        elif data.ndim == 3:
            self.is_3d = True # It's a 3D volume
        else:
            raise ValueError(f"Input data must be 2D or 3D, but got {data.ndim} dimensions.")
        
        self.source_data = data
        self.target_data = None
        self.source_bands_raw = None
        self.target_bands_raw = None
        self.radii = None
        self.metrics = None
        print(f"✓ FrequencyAnalyzer initialized for {'3D volume' if self.is_3d else '2D image'}.")

    # --- Public Methods ---

    def degrade(self, degradation_type='blur_only', **kwargs):
        """Creates a degraded version of the source data."""
        print(f"✓ Applying '{degradation_type}' degradation...")
        if self.is_3d:
            if degradation_type == 'blur_only':
                self.target_data = self._degrade_blur_3d(self.source_data, **kwargs)
            elif degradation_type == 'downsample_only':
                self.target_data = self._degrade_downsample_3d(self.source_data, **kwargs)
            elif degradation_type == 'blur_then_downsample':
                self.target_data = self._degrade_blur_then_downsample_3d(self.source_data, **kwargs)
        else: # 2D
            if degradation_type == 'blur_only':
                self.target_data = self._degrade_blur_2d(self.source_data, **kwargs)

    def analyze(self, radii):
        """Separates data into frequency bands."""
        if self.target_data is None: raise RuntimeError("Run .degrade() before .analyze().")
        self.radii = sorted(radii)
        print(f"✓ Separating data into {len(self.radii)} frequency bands...")
        
        separator_func = self._separate_frequency_bands_3d if self.is_3d else self._separate_frequency_bands_2d
        self.source_bands_raw = separator_func(self.source_data, self.radii)
        self.target_bands_raw = separator_func(self.target_data, self.radii)

    def calculate_metrics(self):
        """Calculates metrics and returns a Pandas DataFrame."""
        if self.source_bands_raw is None: raise RuntimeError("Run .analyze() before .calculate_metrics().")
        print("✓ Calculating numerical metrics...")
        
        metric_calculator = self._calculate_metrics_3d if self.is_3d else self._calculate_metrics_2d
        self.metrics = metric_calculator()
        return self.metrics

    def report(self):
        """Prints a formatted report of the calculated metrics."""
        if self.metrics is None: self.calculate_metrics()
        
        title = f" {'3D' if self.is_3d else '2D'} NUMERICAL FREQUENCY DIFFERENCE ANALYSIS "
        print("\n" + f"{title:=^75}");
        # Set pandas display options for cleaner output
        pd.options.display.float_format = '{:,.3f}'.format
        print(self.metrics)
        print("=" * 75 + "\n")

    def plot(self):
        """Generates a visual comparison plot."""
        if self.target_data is None: raise RuntimeError("Run .degrade() and .analyze() before plotting.")
        print("✓ Generating visual plot...")

        plotter_func = self._plot_3d if self.is_3d else self._plot_2d
        plotter_func()

    # --- Private Helper Methods (Degradation) ---

    def _degrade_blur_2d(self, data, blur_kernel_size=(11, 11)):
        return cv2.GaussianBlur(data, blur_kernel_size, 0)
    
    def _degrade_blur_3d(self, data, sigma=2.0):
        return gaussian_filter(data, sigma=sigma)

    def _degrade_downsample_3d(self, data, zoom_factor=0.5):
        downsampled = zoom(data, zoom_factor, order=1)
        upsampled = zoom(downsampled, 1/zoom_factor, order=1)
        pad = [(0, s - u) for s, u in zip(data.shape, upsampled.shape)]
        return np.pad(upsampled, pad, mode='edge')

    def _degrade_blur_then_downsample_3d(self, data, sigma=2.0, zoom_factor=0.5):
        blurred = self._degrade_blur_3d(data, sigma)
        return self._degrade_downsample_3d(blurred, zoom_factor)

    # --- Private Helper Methods (Analysis) ---

    def _separate_frequency_bands_2d(self, data, radii):
        gray = data if data.ndim == 2 else cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        band_images, last_r = [], 0
        for r in radii:
            mask = np.zeros((rows, cols, 2), np.float32)
            cv2.circle(mask, (ccol, crow), r, (1, 1), -1)
            if last_r > 0:
                inner_mask = np.zeros((rows, cols, 2), np.float32)
                cv2.circle(inner_mask, (ccol, crow), last_r, (1, 1), -1)
                band_mask = mask - inner_mask
            else: band_mask = mask
            fshift_band = dft_shift * band_mask
            f_ishift_band = np.fft.ifftshift(fshift_band)
            img_back = cv2.idft(f_ishift_band)
            band_images.append(cv2.magnitude(img_back[:,:,0], img_back[:,:,1]))
            last_r = r
        return band_images

    def _separate_frequency_bands_3d(self, data, radii):
        dft = np.fft.fftn(data)
        dft_shift = np.fft.fftshift(dft)
        d, h, w = data.shape
        cz, cy, cx = d//2, h//2, w//2
        z, y, x = np.ogrid[-cz:d-cz, -cy:h-cy, -cx:w-cx]
        band_volumes, last_r_sq = [], 0
        for r in radii:
            r_sq = r**2
            mask_3d = (x*x + y*y + z*z) <= r_sq
            if last_r_sq > 0:
                inner_mask_3d = (x*x + y*y + z*z) <= last_r_sq
                band_mask = mask_3d & ~inner_mask_3d
            else: band_mask = mask_3d
            fshift_band = dft_shift * band_mask
            f_ishift_band = np.fft.ifftshift(fshift_band)
            volume_back = np.fft.ifftn(f_ishift_band)
            band_volumes.append(np.abs(volume_back))
            last_r_sq = r_sq
        return band_volumes

    # --- Private Helper Methods (Metrics & Plotting) ---

    def _calculate_metrics_2d(self):
        results, last_r = [], 0
        for i, r in enumerate(self.radii):
            s_band, t_band = self.source_bands_raw[i], self.target_bands_raw[i]
            s_norm, t_norm = self._normalize_array(s_band), self._normalize_array(t_band)
            mae = np.mean(np.abs(s_norm - t_norm)) * 255
            ssim_val = ssim(s_norm, t_norm, data_range=1.0)
            energy_loss = self._calculate_energy_loss(s_band, t_band)
            results.append({'Band': f"{last_r}-{r}px", 'MAE': mae, 'SSIM': ssim_val, 'Energy Loss (%)': energy_loss})
            last_r = r
        
        source_gray = self.source_data if self.source_data.ndim == 2 else cv2.cvtColor(self.source_data, cv2.COLOR_BGR2GRAY)
        target_gray = self.target_data if self.target_data.ndim == 2 else cv2.cvtColor(self.target_data, cv2.COLOR_BGR2GRAY)
        total_loss = self._calculate_energy_loss(source_gray - np.mean(source_gray), target_gray - np.mean(target_gray))
        results.append({'Band': 'TOTAL AC', 'MAE': np.nan, 'SSIM': np.nan, 'Energy Loss (%)': total_loss})
        
        df = pd.DataFrame(results).set_index('Band')
        df['Trend'] = df['Energy Loss (%)'].apply(lambda x: '↓' if x > 0 else '↑')
        return df
    
    def _calculate_metrics_3d(self):
        results, last_r = [], 0
        for i, r in enumerate(self.radii):
            s_band, t_band = self.source_bands_raw[i], self.target_bands_raw[i]
            s_norm, t_norm = self._normalize_array(s_band), self._normalize_array(t_band)
            mae = np.mean(np.abs(s_norm - t_norm)) * 255
            ssim_val = ssim(s_norm, t_norm, data_range=1.0)
            energy_loss = self._calculate_energy_loss(s_band, t_band)
            results.append({'Band': f"{last_r}-{r}vx", 'MAE': mae, 'SSIM': ssim_val, 'Energy Loss (%)': energy_loss})
            last_r = r

        total_loss = self._calculate_energy_loss(self.source_data - np.mean(self.source_data), self.target_data - np.mean(self.target_data))
        results.append({'Band': 'TOTAL AC', 'MAE': np.nan, 'SSIM': np.nan, 'Energy Loss (%)': total_loss})
        
        df = pd.DataFrame(results).set_index('Band')
        df['Trend'] = df['Energy Loss (%)'].apply(lambda x: '↓' if x > 0 else '↑')
        return df

    def _plot_2d(self):
        num_rows = len(self.radii) + 1
        plt.figure(figsize=(15, num_rows * 4.5)); plt.suptitle("2D Visual Frequency Comparison", fontsize=20)
        
        s_rgb = cv2.cvtColor(self.source_data, cv2.COLOR_BGR2RGB)
        t_rgb = cv2.cvtColor(self.target_data, cv2.COLOR_BGR2RGB)
        plt.subplot(num_rows, 3, 1); plt.title("1. Source (Original)"); plt.imshow(s_rgb); plt.axis('off')
        plt.subplot(num_rows, 3, 2); plt.title("2. Target (Degraded)"); plt.imshow(t_rgb); plt.axis('off')
        plt.subplot(num_rows, 3, 3); plt.title("3. Difference (1 vs 2)"); plt.imshow(cv2.absdiff(s_rgb, t_rgb)); plt.axis('off')

        last_r = 0
        for i, r in enumerate(self.radii):
            s_band_norm, t_band_norm = self._normalize_array(self.source_bands_raw[i]), self._normalize_array(self.target_bands_raw[i])
            diff_band = np.abs(s_band_norm - t_band_norm)
            row, title = (i+1)*3, f"Freq. Band ({last_r}-{r}px)"
            plt.subplot(num_rows, 3, row + 1); plt.title(f"Source: {title}"); plt.imshow(s_band_norm, cmap='gray'); plt.axis('off')
            plt.subplot(num_rows, 3, row + 2); plt.title(f"Target: {title}"); plt.imshow(t_band_norm, cmap='gray'); plt.axis('off')
            plt.subplot(num_rows, 3, row + 3); plt.title(f"Difference: {title}"); plt.imshow(diff_band, cmap='gray'); plt.axis('off')
            last_r = r
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show()

    def _plot_3d(self):
        num_rows = len(self.radii) + 1
        plt.figure(figsize=(12, num_rows * 3.5)); plt.suptitle("3D Frequency Comparison (Central Axial Slice)", fontsize=20)
        slice_idx = self.source_data.shape[2] // 2
        
        s_slice, t_slice = self._normalize_array(self.source_data[:,:,slice_idx]), self._normalize_array(self.target_data[:,:,slice_idx])
        plt.subplot(num_rows, 3, 1); plt.title("1. Source (Original)"); plt.imshow(s_slice, cmap='gray'); plt.axis('off')
        plt.subplot(num_rows, 3, 2); plt.title("2. Target (Degraded)"); plt.imshow(t_slice, cmap='gray'); plt.axis('off')
        plt.subplot(num_rows, 3, 3); plt.title("3. Difference (1 vs 2)"); plt.imshow(np.abs(s_slice-t_slice), cmap='gray'); plt.axis('off')

        last_r = 0
        for i, r in enumerate(self.radii):
            s_band_slice, t_band_slice = self._normalize_array(self.source_bands_raw[i][:,:,slice_idx]), self._normalize_array(self.target_bands_raw[i][:,:,slice_idx])
            row, title = (i + 1) * 3, f"Freq. Band ({last_r}-{r}vx)"
            plt.subplot(num_rows, 3, row + 1); plt.title(f"Source: {title}"); plt.imshow(s_band_slice, cmap='gray'); plt.axis('off')
            plt.subplot(num_rows, 3, row + 2); plt.title(f"Target: {title}"); plt.imshow(t_band_slice, cmap='gray'); plt.axis('off')
            plt.subplot(num_rows, 3, row + 3); plt.title(f"Difference: {title}"); plt.imshow(np.abs(s_band_slice - t_band_slice), cmap='gray'); plt.axis('off')
            last_r = r
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show()

    # --- Static Utility Methods ---
    
    @staticmethod
    def _normalize_array(arr):
        min_val, max_val = arr.min(), arr.max()
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

    @staticmethod
    def _calculate_energy_loss(source, target):
        s_64, t_64 = source.astype(np.float64), target.astype(np.float64)
        energy_s = np.sum(s_64**2)
        energy_t = np.sum(t_64**2)
        return 100 * (energy_s - energy_t) / energy_s if energy_s > 0 else 0.0

# ==============================================================================
# 3. EXAMPLE USAGE
# ==============================================================================

def test():
    # --- Example 1: 2D Image Analysis ---
    print("="*40); print("--- Running 2D Image Example ---"); print("="*40)
    try:
        IMAGE_URL = "https://picsum.photos/1024/768"
        response = requests.get(IMAGE_URL, headers={'User-Agent': 'Mozilla/5.0'}); response.raise_for_status()
        pil_image = Image.open(io.BytesIO(response.content))
        source_2d_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        analyzer_2d = FrequencyAnalyzer(source_2d_image)
        analyzer_2d.degrade(degradation_type='blur_only', blur_kernel_size=(15, 15))
        analyzer_2d.analyze(radii=[15, 40, 100, 250])
        analyzer_2d.report()
        analyzer_2d.plot()
        
    except Exception as e:
        print(f"❌ 2D Example Failed: {e}")


    # --- Example 2: 3D NIfTI Volume Analysis ---
    print("\n" + "="*40); print("--- Running 3D NIfTI Volume Example ---"); print("="*40)
    try:
        def create_synthetic_3d_volume(shape=(96, 96, 96)):
            x, y, z = np.ogrid[-1:1:shape[0]*1j, -1:1:shape[1]*1j, -1:1:shape[2]*1j]
            volume = np.zeros(shape); volume[x**2 + y**2 + z**2 <= 0.7**2] = 1.0
            volume[(x-0.4)**2 + (y-0.4)**2 + z**2 <= 0.2**2] = 1.8
            return volume + np.random.normal(0, 0.05, size=shape)

        source_3d_volume = create_synthetic_3d_volume()

        analyzer_3d = FrequencyAnalyzer(source_3d_volume)
        analyzer_3d.degrade(degradation_type='blur_only', sigma=2.0)
        analyzer_3d.analyze(radii=[8, 16, 32])
        analyzer_3d.report()
        analyzer_3d.plot()

    except Exception as e:
        print(f"❌ 3D Example Failed: {e}")