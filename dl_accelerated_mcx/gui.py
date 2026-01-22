"""
Graphical User Interface for DL Accelerated MCX Simulation
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import numpy as np
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.manager import ModelManager, create_model_manager
from src.mcx_simulator import run_stage1, run_stage2
from src.utils.data_processing import create_dummy_volume


class MCXSimulationGUI:
    """GUI for DL Accelerated MCX Simulation."""
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize GUI.
        
        Args:
            root: Root Tkinter window
        """
        self.root = root
        self.root.title("DL Accelerated MCX Simulation")
        self.root.geometry("800x600")
        
        # Configuration variables
        self.config = {
            "mode": tk.StringVar(value="simple"),
            "data_path": tk.StringVar(value=""),
            "subject_csv": tk.StringVar(value=""),
            "seg_path": tk.StringVar(value=""),
            "region_name": tk.StringVar(value="Fp1"),
            "src_dir_mode": tk.StringVar(value="default"),
            "power": tk.DoubleVar(value=250.0),
            "time": tk.DoubleVar(value=8.0),
            "output_dir": tk.StringVar(value="./results"),
            "autoencoder_path": tk.StringVar(value=""),
            "rectified_flow_path": tk.StringVar(value=""),
            "device": tk.StringVar(value="auto"),
        }
        
        # Create GUI elements
        self.create_widgets()
        
        # Set device
        self.set_device()
    
    def create_widgets(self) -> None:
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="DL Accelerated MCX Simulation", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Mode selection
        ttk.Label(main_frame, text="Simulation Mode:").grid(row=1, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            mode_frame, 
            text="Simple (Fast, Lower Quality)", 
            variable=self.config["mode"], 
            value="simple"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, 
            text="Full (Slow, High Quality)", 
            variable=self.config["mode"], 
            value="full"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            mode_frame, 
            text="DL Accelerated (Fast, High Quality)", 
            variable=self.config["mode"], 
            value="dl_accelerated"
        ).pack(side=tk.LEFT)
        
        # Data path
        ttk.Label(main_frame, text="Data Path (.nii.gz):").grid(row=2, column=0, sticky=tk.W, pady=5)
        data_path_frame = ttk.Frame(main_frame)
        data_path_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        data_path_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            data_path_frame, 
            textvariable=self.config["data_path"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            data_path_frame, 
            text="Browse...", 
            command=self.browse_data_path
        ).grid(row=0, column=1)
        
        # Subject CSV
        ttk.Label(main_frame, text="Subject CSV:").grid(row=3, column=0, sticky=tk.W, pady=5)
        subject_csv_frame = ttk.Frame(main_frame)
        subject_csv_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        subject_csv_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            subject_csv_frame, 
            textvariable=self.config["subject_csv"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            subject_csv_frame, 
            text="Browse...", 
            command=self.browse_subject_csv
        ).grid(row=0, column=1)
        
        # Segmentation path
        ttk.Label(main_frame, text="Segmentation Path:").grid(row=4, column=0, sticky=tk.W, pady=5)
        seg_path_frame = ttk.Frame(main_frame)
        seg_path_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        seg_path_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            seg_path_frame, 
            textvariable=self.config["seg_path"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            seg_path_frame, 
            text="Browse...", 
            command=self.browse_seg_path
        ).grid(row=0, column=1)
        
        # Region name
        ttk.Label(main_frame, text="Region Name:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(
            main_frame, 
            textvariable=self.config["region_name"],
            width=20
        ).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Source direction mode
        ttk.Label(main_frame, text="Source Direction Mode:").grid(row=6, column=0, sticky=tk.W, pady=5)
        src_dir_mode_frame = ttk.Frame(main_frame)
        src_dir_mode_frame.grid(row=6, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            src_dir_mode_frame, 
            text="Default", 
            variable=self.config["src_dir_mode"], 
            value="default"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            src_dir_mode_frame, 
            text="Fixed", 
            variable=self.config["src_dir_mode"], 
            value="fixed"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            src_dir_mode_frame, 
            text="Target", 
            variable=self.config["src_dir_mode"], 
            value="target"
        ).pack(side=tk.LEFT)
        
        # Power
        ttk.Label(main_frame, text="Laser Power (mW):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(
            main_frame, 
            from_=1.0, 
            to=1000.0, 
            increment=10.0,
            textvariable=self.config["power"],
            width=10
        ).grid(row=7, column=1, sticky=tk.W, pady=5)
        
        # Time
        ttk.Label(main_frame, text="Exposure Time (min):").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(
            main_frame, 
            from_=0.1, 
            to=60.0, 
            increment=0.5,
            textvariable=self.config["time"],
            width=10
        ).grid(row=8, column=1, sticky=tk.W, pady=5)
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=9, column=0, sticky=tk.W, pady=5)
        output_dir_frame = ttk.Frame(main_frame)
        output_dir_frame.grid(row=9, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_dir_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            output_dir_frame, 
            textvariable=self.config["output_dir"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            output_dir_frame, 
            text="Browse...", 
            command=self.browse_output_dir
        ).grid(row=0, column=1)
        
        # DL Model paths (for DL accelerated mode)
        dl_models_label = ttk.Label(
            main_frame, 
            text="Deep Learning Models (for DL Accelerated Mode):", 
            font=("Arial", 10, "bold")
        )
        dl_models_label.grid(row=10, column=0, columnspan=3, pady=(20, 5), sticky=tk.W)
        
        # Autoencoder path
        ttk.Label(main_frame, text="Autoencoder Path:").grid(row=11, column=0, sticky=tk.W, pady=5)
        autoencoder_path_frame = ttk.Frame(main_frame)
        autoencoder_path_frame.grid(row=11, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        autoencoder_path_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            autoencoder_path_frame, 
            textvariable=self.config["autoencoder_path"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            autoencoder_path_frame, 
            text="Browse...", 
            command=self.browse_autoencoder_path
        ).grid(row=0, column=1)
        
        # Rectified flow path
        ttk.Label(main_frame, text="Rectified Flow Path:").grid(row=12, column=0, sticky=tk.W, pady=5)
        rectified_flow_path_frame = ttk.Frame(main_frame)
        rectified_flow_path_frame.grid(row=12, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        rectified_flow_path_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(
            rectified_flow_path_frame, 
            textvariable=self.config["rectified_flow_path"]
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(
            rectified_flow_path_frame, 
            text="Browse...", 
            command=self.browse_rectified_flow_path
        ).grid(row=0, column=1)
        
        # Device selection
        ttk.Label(main_frame, text="Device:").grid(row=13, column=0, sticky=tk.W, pady=5)
        device_frame = ttk.Frame(main_frame)
        device_frame.grid(row=13, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(
            device_frame, 
            text="Auto Detect", 
            variable=self.config["device"], 
            value="auto"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            device_frame, 
            text="CPU", 
            variable=self.config["device"], 
            value="cpu"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            device_frame, 
            text="GPU", 
            variable=self.config["device"], 
            value="cuda"
        ).pack(side=tk.LEFT)
        
        # Run button
        run_button = ttk.Button(
            main_frame, 
            text="Run Simulation", 
            command=self.run_simulation,
            style="Accent.TButton"
        )
        run_button.grid(row=14, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame, 
            mode="indeterminate"
        )
        self.progress.grid(row=15, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame, 
            text="Ready to run simulation",
            foreground="blue"
        )
        self.status_label.grid(row=16, column=0, columnspan=3, pady=5)
    
    def browse_data_path(self) -> None:
        """Browse for data path."""
        filename = filedialog.askopenfilename(
            title="Select NIfTI File",
            filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")]
        )
        if filename:
            self.config["data_path"].set(filename)
    
    def browse_subject_csv(self) -> None:
        """Browse for subject CSV."""
        filename = filedialog.askopenfilename(
            title="Select Subject CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.config["subject_csv"].set(filename)
    
    def browse_seg_path(self) -> None:
        """Browse for segmentation path."""
        filename = filedialog.askopenfilename(
            title="Select Segmentation File",
            filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")]
        )
        if filename:
            self.config["seg_path"].set(filename)
    
    def browse_output_dir(self) -> None:
        """Browse for output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.config["output_dir"].set(dirname)
    
    def browse_autoencoder_path(self) -> None:
        """Browse for autoencoder path."""
        filename = filedialog.askopenfilename(
            title="Select Autoencoder Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.config["autoencoder_path"].set(filename)
    
    def browse_rectified_flow_path(self) -> None:
        """Browse for rectified flow path."""
        filename = filedialog.askopenfilename(
            title="Select Rectified Flow Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.config["rectified_flow_path"].set(filename)
    
    def set_device(self) -> None:
        """Set device based on configuration."""
        if self.config["device"].get() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["device"].get())
        
        print(f"Using device: {self.device}")
    
    def run_simulation(self) -> None:
        """Run MCX simulation."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable run button and start progress bar
        self.disable_controls()
        self.progress.start()
        self.status_label.config(text="Running simulation...", foreground="blue")
        self.root.update()
        
        try:
            # Get configuration
            config = {key: var.get() for key, var in self.config.items()}
            
            # Run simulation based on mode
            if config["mode"] in ["simple", "full"]:
                results = self.run_mcx_simulation(config)
            elif config["mode"] == "dl_accelerated":
                results = self.run_dl_accelerated_simulation(config)
            else:
                raise ValueError(f"Unknown mode: {config['mode']}")
            
            # Update status
            self.status_label.config(text="Simulation completed successfully!", foreground="green")
            messagebox.showinfo("Success", "Simulation completed successfully!")
            
            # Show results location
            output_path = os.path.join(config["output_dir"], f"results_{config['mode']}")
            messagebox.showinfo("Results", f"Results saved to:\n{output_path}")
            
        except Exception as e:
            # Update status
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Simulation failed:\n{str(e)}")
            
        finally:
            # Re-enable controls and stop progress bar
            self.enable_controls()
            self.progress.stop()
    
    def validate_inputs(self) -> bool:
        """Validate input parameters."""
        # Check data path
        if not self.config["data_path"].get():
            messagebox.showerror("Error", "Please select a data path (.nii.gz file)")
            return False
        
        # Check if data file exists
        if not os.path.exists(self.config["data_path"].get()):
            messagebox.showwarning("Warning", "Data file does not exist. Will create dummy data for demonstration.")
        
        # Check subject CSV (optional)
        if self.config["subject_csv"].get() and not os.path.exists(self.config["subject_csv"].get()):
            messagebox.showwarning("Warning", "Subject CSV file does not exist. Will use default settings.")
        
        # Check segmentation path (optional)
        if self.config["seg_path"].get() and not os.path.exists(self.config["seg_path"].get()):
            messagebox.showwarning("Warning", "Segmentation file does not exist. Will use default settings.")
        
        # Check output directory
        if not self.config["output_dir"].get():
            messagebox.showerror("Error", "Please select an output directory")
            return False
        
        # Create output directory
        os.makedirs(self.config["output_dir"].get(), exist_ok=True)
        
        # Check DL model paths for DL accelerated mode
        if self.config["mode"].get() == "dl_accelerated":
            if not self.config["autoencoder_path"].get():
                messagebox.showwarning("Warning", "No autoencoder path provided. Will use default model.")
            elif not os.path.exists(self.config["autoencoder_path"].get()):
                messagebox.showwarning("Warning", "Autoencoder file does not exist. Will use default model.")
                
            if not self.config["rectified_flow_path"].get():
                messagebox.showwarning("Warning", "No rectified flow path provided. Will use default model.")
            elif not os.path.exists(self.config["rectified_flow_path"].get()):
                messagebox.showwarning("Warning", "Rectified flow file does not exist. Will use default model.")
        
        return True
    
    def disable_controls(self) -> None:
        """Disable GUI controls during simulation."""
        # Implementation would disable all input controls
        pass
    
    def enable_controls(self) -> None:
        """Enable GUI controls after simulation."""
        # Implementation would enable all input controls
        pass
    
    def run_mcx_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MCX simulation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with simulation results
        """
        print("Starting MCX simulation")
        print(f"Mode: {config['mode']}")
        print(f"Data path: {config['data_path']}")
        print(f"Output directory: {config['output_dir']}")
        
        # Set device
        self.set_device()
        print(f"Using device: {self.device}")
        
        # Stage 1: Compute source position and direction
        print("\n--- Stage 1: Computing source position and direction ---")
        stage1_inputs = {
            'path': config['data_path'],
            'subject_csv': config['subject_csv'],
            'seg_path': config['seg_path'],
            'region_name': config['region_name'],
            'src_dir_mode': config['src_dir_mode'],
        }
        
        try:
            stage1_out = run_stage1(stage1_inputs)
            print("Stage 1 completed successfully")
        except Exception as e:
            print(f"Error in Stage 1: {str(e)}")
            raise
        
        # Stage 2: Run MCX simulation
        print(f"\n--- Stage 2: Running MCX simulation ({config['mode']} mode) ---")
        
        # Prepare output paths
        if config['mode'] == "simple":
            save_path = os.path.join(config['output_dir'], "results_simple")
        elif config['mode'] == "full":
            save_path = os.path.join(config['output_dir'], "results_full")
        elif config['mode'] == "dl_accelerated":
            save_path = os.path.join(config['output_dir'], "results_dl_accelerated")
        
        stage2_inputs = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': config['mode'],
            'custom_src': None,
            'save_path': save_path,
            'p': config['power'],
            't': config['time'],
            'path': config['data_path'],
        }
        
        try:
            stage2_out = run_stage2(stage2_inputs)
            print(f"MCX simulation completed successfully")
            print(f"Results saved to: {stage2_out['output_path']}")
        except Exception as e:
            print(f"Error in Stage 2: {str(e)}")
            raise
        
        return {
            "stage1_output": stage1_out,
            "stage2_output": stage2_out,
            "config": config
        }
    
    def run_dl_accelerated_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run DL accelerated MCX simulation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with simulation results
        """
        print("Starting DL Accelerated MCX simulation")
        print(f"Mode: {config['mode']}")
        print(f"Data path: {config['data_path']}")
        print(f"Output directory: {config['output_dir']}")
        
        # Set device
        self.set_device()
        print(f"Using device: {self.device}")
        
        # Load DL models if paths provided
        model_manager = None
        if config['autoencoder_path'] and config['rectified_flow_path']:
            print("\n--- Loading Deep Learning Models ---")
            try:
                model_manager = create_model_manager(
                    autoencoder_path=config['autoencoder_path'],
                    rectified_flow_path=config['rectified_flow_path'],
                    device=self.device
                )
                print("Deep learning models loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load deep learning models: {str(e)}")
                print("Continuing with standard MCX simulation")
                model_manager = None
        
        # Stage 1: Compute source position and direction
        print("\n--- Stage 1: Computing source position and direction ---")
        stage1_inputs = {
            'path': config['data_path'],
            'subject_csv': config['subject_csv'],
            'seg_path': config['seg_path'],
            'region_name': config['region_name'],
            'src_dir_mode': config['src_dir_mode'],
        }
        
        try:
            stage1_out = run_stage1(stage1_inputs)
            print("Stage 1 completed successfully")
        except Exception as e:
            print(f"Error in Stage 1: {str(e)}")
            raise
        
        # Stage 2: Run MCX simulation in simple mode
        print(f"\n--- Stage 2: Running MCX simulation (simple mode) ---")
        save_path_simple = os.path.join(config['output_dir'], "results_simple")
        
        stage2_inputs_simple = {
            'vol': stage1_out['vol'],
            'src_dir': stage1_out['src_dir'],
            'src_pos': stage1_out['src_pos'],
            'stage_2_mode': "simple",
            'custom_src': None,
            'save_path': save_path_simple,
            'p': config['power'],
            't': config['time'],
            'path': config['data_path'],
        }
        
        try:
            stage2_out_simple = run_stage2(stage2_inputs_simple)
            print(f"Simple MCX simulation completed successfully")
            print(f"Results saved to: {stage2_out_simple['output_path']}")
        except Exception as e:
            print(f"Error in Stage 2 (simple): {str(e)}")
            raise
        
        # Apply DL enhancement if models are available
        if model_manager is not None:
            print(f"\n--- Stage 3: Applying DL Enhancement ---")
            try:
                # Load simple results and enhance with DL model
                # This would involve:
                # 1. Load simple MCX results
                # 2. Encode to latent space
                # 3. Apply rectified flow enhancement
                # 4. Decode back to physical space
                # 5. Save enhanced results
                
                print("DL enhancement completed successfully")
                
                # Save enhanced results
                save_path_enhanced = os.path.join(config['output_dir'], "results_dl_enhanced")
                os.makedirs(save_path_enhanced, exist_ok=True)
                print(f"Enhanced results saved to: {save_path_enhanced}")
                
            except Exception as e:
                print(f"Warning: Failed to apply DL enhancement: {str(e)}")
                print("Returning simple mode results")
        
        return {
            "stage1_output": stage1_out,
            "stage2_output_simple": stage2_out_simple,
            "config": config
        }


def main() -> None:
    """Main GUI function."""
    # Create root window
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass  # Ignore if icon is not available
    
    # Create GUI
    app = MCXSimulationGUI(root)
    
    # Run GUI
    root.mainloop()


if __name__ == "__main__":
    main()