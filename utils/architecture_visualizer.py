import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import tkinter as tk
from tkinter import filedialog
import warnings

warnings.filterwarnings('ignore')


class EnhancedYOLOArchitectureVisualizer:
    def __init__(self, output_dir: str = "figures/visualization"):
        """
        Initialize visualizer with custom output directory

        Args:
            output_dir: Directory to save visualizations (default: "figures/visualization")
        """
        self.output_dir = output_dir
        self._ensure_output_directory()

        # Enhanced color scheme with better coverage of YOLO modules
        self.colors = {
            'Conv': '#FF6B6B',  # Red - Basic convolution
            'C2f': '#4ECDC4',  # Teal - C2f blocks
            'C2fCIB': '#45B7D1',  # Blue - C2f with CIB
            'C2PSA': '#96CEB4',  # Green - C2 with PSA
            'C3Ghost': '#FECA57',  # Yellow - C3 Ghost
            'C3k2': '#FF9FF3',  # Pink - C3k2
            'A2C2f': '#54A0FF',  # Light Blue - Attention C2f
            'GhostConv': '#FD79A8',  # Light Pink - Ghost convolution
            'nn.Upsample': '#A29BFE',  # Purple - Upsampling
            'Upsample': '#A29BFE',  # Purple - Upsampling (alternative)
            'Concat': '#6C5CE7',  # Dark Purple - Concatenation
            'Detect': '#2D3436',  # Dark Gray - Detection head
            'v10Detect': '#636E72',  # Gray - v10 Detection
            'PSA': '#00B894',  # Green - Position-Sensitive Attention
            'SCDown': '#E17055',  # Orange - Spatial Channel Downsampling
            'SPPF': '#FDCB6E',  # Light Orange - Spatial Pyramid Pooling Fast
            'Input': '#85C1E9',  # Light Blue - Input layer
            'Default': '#DDA0DD'  # Light Purple - Default
        }

        # Module descriptions for tooltips or legends
        self.module_descriptions = {
            'Conv': 'Standard Convolution',
            'C2f': 'C2f Block (Fast)',
            'C2fCIB': 'C2f with Contextual Information Block',
            'C2PSA': 'C2 with Position-Sensitive Attention',
            'C3Ghost': 'C3 with Ghost Convolution',
            'C3k2': 'C3 with Kernel Size 2',
            'A2C2f': 'Attention-based C2f',
            'GhostConv': 'Ghost Convolution',
            'nn.Upsample': 'Upsampling Layer',
            'Upsample': 'Upsampling Layer',
            'Concat': 'Concatenation',
            'Detect': 'Detection Head',
            'v10Detect': 'YOLOv10 Detection Head',
            'PSA': 'Position-Sensitive Attention',
            'SCDown': 'Spatial Channel Downsampling',
            'SPPF': 'Spatial Pyramid Pooling Fast',
            'Input': 'Input Layer'
        }

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✓ Output directory ready: {self.output_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not create output directory {self.output_dir}: {e}")
            print("⚠ Files will be saved in current directory")
            self.output_dir = "."

    def get_save_path(self, filename: str) -> str:
        """Generate full save path with output directory"""
        return os.path.join(self.output_dir, filename)

    def select_yaml_file(self):
        """File dialog to select YAML configuration file"""
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="Select YOLO YAML Configuration File",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )

        root.destroy()
        return file_path

    def load_yaml_file(self, file_path: str) -> Dict:
        """Load YAML configuration file with enhanced error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                print(f"✓ Successfully loaded YAML: {os.path.basename(file_path)}")
                return config
        except FileNotFoundError:
            print(f"✗ Error: File not found - {file_path}")
        except yaml.YAMLError as e:
            print(f"✗ Error parsing YAML: {e}")
        except Exception as e:
            print(f"✗ Unexpected error loading file: {e}")
        return {}

    def parse_yaml(self, yaml_content: str) -> Dict:
        """Parse YAML content and extract architecture info"""
        try:
            config = yaml.safe_load(yaml_content)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return {}

    def get_layer_color(self, layer_type: str) -> str:
        """Get color for layer type with enhanced matching"""
        # Handle nested module names
        if '.' in layer_type:
            layer_type = layer_type.split('.')[-1]

        # Exact match first
        if layer_type in self.colors:
            return self.colors[layer_type]

        # Partial match for complex module names
        for key in self.colors:
            if key in layer_type or layer_type in key:
                return self.colors[key]

        return self.colors['Default']

    def extract_layers(self, config: Dict) -> List[Dict]:
        """Extract layer information from config with enhanced parsing"""
        layers = []

        # Add input layer
        layers.append({
            'id': -1,
            'name': 'Input',
            'type': 'Input',
            'from': [],
            'args': ['Image'],
            'section': 'input'
        })

        # Process backbone
        if 'backbone' in config:
            for i, layer in enumerate(config['backbone']):
                if isinstance(layer, list) and len(layer) >= 3:
                    layers.append({
                        'id': i,
                        'name': f"{layer[2]}_{i}",
                        'type': layer[2],
                        'from': [layer[0]] if isinstance(layer[0], int) else layer[0],
                        'args': layer[3] if len(layer) > 3 else [],
                        'repeats': layer[1] if len(layer) > 1 else 1,
                        'section': 'backbone'
                    })

        # Process head
        if 'head' in config:
            backbone_len = len(config['backbone']) if 'backbone' in config else 0
            for i, layer in enumerate(config['head']):
                if isinstance(layer, list) and len(layer) >= 3:
                    layer_id = backbone_len + i
                    layers.append({
                        'id': layer_id,
                        'name': f"{layer[2]}_{layer_id}",
                        'type': layer[2],
                        'from': [layer[0]] if isinstance(layer[0], int) else layer[0],
                        'args': layer[3] if len(layer) > 3 else [],
                        'repeats': layer[1] if len(layer) > 1 else 1,
                        'section': 'head'
                    })

        return layers

    def get_layer_position(self, layer_id: int, layers: List[Dict]) -> Tuple[float, float]:
        """Calculate position for each layer with better spacing"""
        section_positions = {'input': 0, 'backbone': 1, 'head': 2}

        layer = next((l for l in layers if l['id'] == layer_id), None)
        if not layer:
            return (0, 0)

        section = layer['section']
        section_layers = [l for l in layers if l['section'] == section]
        layer_index = next((i for i, l in enumerate(section_layers) if l['id'] == layer_id), 0)

        x = section_positions[section] * 5  # Increased spacing
        y = layer_index * 2  # Increased vertical spacing

        return (x, y)

    def create_layer_box(self, ax, layer: Dict, position: Tuple[float, float]) -> patches.Rectangle:
        """Create enhanced visual box for a layer"""
        x, y = position
        layer_type = layer['type']

        # Handle nested module names
        if '.' in layer_type:
            display_type = layer_type.split('.')[-1]
        else:
            display_type = layer_type

        color = self.get_layer_color(layer_type)

        # Create rounded rectangle with enhanced styling
        box = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)

        # Enhanced text with layer information
        main_text = f"{display_type}"
        if layer.get('repeats', 1) > 1:
            main_text += f" ×{layer['repeats']}"

        ax.text(x, y + 0.1, main_text,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if self._is_dark_color(color) else 'black')

        # Layer ID
        ax.text(x, y - 0.2, f"[{layer['id']}]",
                ha='center', va='center', fontsize=8,
                color='white' if self._is_dark_color(color) else 'black')

        # Add arguments info if available
        if layer.get('args') and isinstance(layer['args'], list) and len(layer['args']) > 0:
            args_text = str(layer['args'][0]) if layer['args'][0] is not None else ''
            if args_text and len(args_text) < 10:
                ax.text(x + 1.2, y, f"({args_text})",
                        ha='left', va='center', fontsize=7, style='italic',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        return box

    def _is_dark_color(self, color: str) -> bool:
        """Check if color is dark to determine text color"""
        dark_colors = ['#2D3436', '#636E72', '#6C5CE7']
        return color in dark_colors

    def draw_connections(self, ax, layers: List[Dict]):
        """Draw enhanced connections between layers"""
        for layer in layers:
            if layer['id'] == -1:  # Skip input layer connections
                continue

            from_layers = layer['from']
            if not isinstance(from_layers, list):
                from_layers = [from_layers]

            current_pos = self.get_layer_position(layer['id'], layers)

            for from_id in from_layers:
                if isinstance(from_id, list):  # Handle multiple inputs
                    for sub_id in from_id:
                        if isinstance(sub_id, int):
                            actual_from_id = sub_id if sub_id >= 0 else layer['id'] + sub_id
                            from_pos = self.get_layer_position(actual_from_id, layers)
                            self.draw_arrow(ax, from_pos, current_pos, 'concat')
                else:
                    actual_from_id = from_id if from_id >= 0 else layer['id'] + from_id
                    from_pos = self.get_layer_position(actual_from_id, layers)
                    arrow_type = 'sequential' if from_id == -1 else 'skip'
                    self.draw_arrow(ax, from_pos, current_pos, arrow_type)

    def draw_arrow(self, ax, from_pos: Tuple[float, float], to_pos: Tuple[float, float],
                   arrow_type: str = 'sequential'):
        """Draw enhanced arrow between two positions"""
        if arrow_type == 'sequential':
            arrow = ConnectionPatch(
                from_pos, to_pos, "data", "data",
                arrowstyle="->", shrinkA=30, shrinkB=30,
                mutation_scale=20, fc="black", ec="black", alpha=0.8, linewidth=2
            )
        elif arrow_type == 'concat':
            arrow = ConnectionPatch(
                from_pos, to_pos, "data", "data",
                arrowstyle="->", shrinkA=30, shrinkB=30,
                mutation_scale=15, fc="#E74C3C", ec="#E74C3C", alpha=0.7,
                connectionstyle="arc3,rad=0.3", linewidth=2
            )
        else:  # skip connection
            arrow = ConnectionPatch(
                from_pos, to_pos, "data", "data",
                arrowstyle="->", shrinkA=30, shrinkB=30,
                mutation_scale=15, fc="#3498DB", ec="#3498DB", alpha=0.7,
                connectionstyle="arc3,rad=0.2", linewidth=2
            )

        ax.add_patch(arrow)

    def add_enhanced_legend(self, ax, layers: List[Dict]):
        """Add comprehensive legend with layer statistics"""
        # Get unique layer types used in the model
        used_types = set()
        for layer in layers:
            layer_type = layer['type']
            if '.' in layer_type:
                layer_type = layer_type.split('.')[-1]
            used_types.add(layer_type)

        # Create legend
        legend_elements = []
        for layer_type in sorted(used_types):
            color = self.get_layer_color(layer_type)
            description = self.module_descriptions.get(layer_type, layer_type)
            legend_elements.append(patches.Patch(color=color, label=f"{layer_type}: {description}"))

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=10, title="Layer Types", title_fontsize=12)

    def add_model_info(self, ax, config: Dict, layers: List[Dict]):
        """Add model information panel"""
        info_text = []

        # Basic info
        if 'nc' in config:
            info_text.append(f"Classes: {config['nc']}")

        # Layer counts
        backbone_count = len([l for l in layers if l['section'] == 'backbone'])
        head_count = len([l for l in layers if l['section'] == 'head'])
        total_count = backbone_count + head_count

        info_text.extend([
            f"Total Layers: {total_count}",
            f"Backbone: {backbone_count}",
            f"Head: {head_count}"
        ])

        # Scale information
        if 'scales' in config:
            scales = list(config['scales'].keys())
            info_text.append(f"Scales: {', '.join(scales)}")

        # Display info
        info_str = '\n'.join(info_text)
        ax.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                                   facecolor='lightblue', alpha=0.8))

    def visualize_architecture(self, yaml_content: str = None, config: Dict = None,
                               title: str = "YOLO Architecture", save_path: str = None):
        """Main visualization function with enhanced features"""
        if config is None:
            if yaml_content is None:
                print("No YAML content or config provided")
                return None, None
            config = self.parse_yaml(yaml_content)

        if not config:
            print("Failed to parse configuration")
            return None, None

        layers = self.extract_layers(config)
        if not layers:
            print("No layers found in configuration")
            return None, None

        # Create figure with better proportions
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))

        # Draw layers
        for layer in layers:
            position = self.get_layer_position(layer['id'], layers)
            self.create_layer_box(ax, layer, position)

        # Draw connections
        self.draw_connections(ax, layers)

        # Add section labels with enhanced styling
        sections = ['INPUT', 'BACKBONE', 'HEAD']
        section_colors = ['#E3F2FD', '#FFF3E0', '#F3E5F5']
        for i, (section, color) in enumerate(zip(sections, section_colors)):
            x_pos = i * 5
            ax.text(x_pos, -1.5, section, ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))

        # Add enhanced legend and model info
        self.add_enhanced_legend(ax, layers)
        self.add_model_info(ax, config, layers)

        # Set limits and styling
        max_y = max([self.get_layer_position(l['id'], layers)[1] for l in layers]) if layers else 0
        ax.set_xlim(-2, 12)
        ax.set_ylim(-3, max_y + 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=30)

        plt.tight_layout()

        # Save if requested
        if save_path:
            # Use custom output directory
            full_save_path = self.get_save_path(save_path)
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"✓ Visualization saved to: {full_save_path}")

        return fig, ax

    def visualize_from_file(self, file_path: str = None, title: str = None):
        """Visualize architecture from YAML file"""
        if file_path is None:
            print("Please select a YOLO YAML configuration file...")
            file_path = self.select_yaml_file()

        if not file_path:
            print("No file selected.")
            return None, None

        config = self.load_yaml_file(file_path)
        if not config:
            return None, None

        # Generate title from filename if not provided
        if title is None:
            title = f"YOLO Architecture - {os.path.splitext(os.path.basename(file_path))[0]}"

        # Generate save path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = f"{base_name}_architecture_visualization.png"

        return self.visualize_architecture(config=config, title=title, save_path=save_path)

    def compare_architectures(self, file_paths: List[str] = None, titles: List[str] = None):
        """Compare multiple architectures side by side"""
        if file_paths is None:
            print("Select multiple YAML files for comparison...")
            file_paths = []
            while True:
                file_path = self.select_yaml_file()
                if not file_path:
                    break
                file_paths.append(file_path)

                continue_selection = input("Select another file? (y/n): ").lower()
                if continue_selection != 'y':
                    break

        if len(file_paths) < 2:
            print("Need at least 2 files for comparison")
            return None, None

        configs = []
        for file_path in file_paths:
            config = self.load_yaml_file(file_path)
            if config:
                configs.append(config)

        if not configs:
            print("No valid configurations loaded")
            return None, None

        # Generate titles if not provided
        if titles is None:
            titles = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]

        # Create comparison visualization
        n_models = len(configs)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 12))

        if n_models == 1:
            axes = [axes]

        for i, (config, title) in enumerate(zip(configs, titles)):
            layers = self.extract_layers(config)
            if not layers:
                continue

            ax = axes[i]

            # Draw layers with adjusted positions for comparison
            for layer in layers:
                position = self.get_layer_position(layer['id'], layers)
                adjusted_pos = (position[0] * 0.7, position[1])
                self.create_layer_box(ax, layer, adjusted_pos)

            # Draw connections with adjusted positions
            for layer in layers:
                if layer['id'] == -1:
                    continue

                from_layers = layer['from']
                if not isinstance(from_layers, list):
                    from_layers = [from_layers]

                current_pos = self.get_layer_position(layer['id'], layers)
                current_pos = (current_pos[0] * 0.7, current_pos[1])

                for from_id in from_layers:
                    if isinstance(from_id, list):
                        for sub_id in from_id:
                            if isinstance(sub_id, int):
                                actual_from_id = sub_id if sub_id >= 0 else layer['id'] + sub_id
                                from_pos = self.get_layer_position(actual_from_id, layers)
                                from_pos = (from_pos[0] * 0.7, from_pos[1])
                                self.draw_arrow(ax, from_pos, current_pos, 'concat')
                    else:
                        actual_from_id = from_id if from_id >= 0 else layer['id'] + from_id
                        from_pos = self.get_layer_position(actual_from_id, layers)
                        from_pos = (from_pos[0] * 0.7, from_pos[1])
                        arrow_type = 'sequential' if from_id == -1 else 'skip'
                        self.draw_arrow(ax, from_pos, current_pos, arrow_type)

            # Styling for each subplot
            max_y = max([self.get_layer_position(l['id'], layers)[1] for l in layers]) if layers else 0
            ax.set_xlim(-1, 8)
            ax.set_ylim(-2, max_y + 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Add common legend
        all_layers = []
        for config in configs:
            all_layers.extend(self.extract_layers(config))

        if all_layers:
            self.add_enhanced_legend(fig.gca(), all_layers)

        plt.tight_layout()

        # Save comparison with custom output directory
        save_path = self.get_save_path("architecture_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✓ Comparison saved to: {save_path}")

        return fig, axes


def main():
    """Enhanced main function with interactive menu"""
    print("Enhanced YOLO Architecture Visualizer")

    # Initialize with custom output directory
    visualizer = EnhancedYOLOArchitectureVisualizer(output_dir="figures/visualization")

    while True:
        print("\nOptions:")
        print("1. Visualize single architecture")
        print("2. Compare multiple architectures")
        print("3. Change output directory")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            print("\nVisualizing single architecture...")
            fig, ax = visualizer.visualize_from_file()
            if fig:
                plt.show()

        elif choice == '2':
            print("\nComparing multiple architectures...")
            fig, axes = visualizer.compare_architectures()
            if fig:
                plt.show()

        elif choice == '3':
            new_dir = input("Enter new output directory path: ").strip()
            if new_dir:
                visualizer.output_dir = new_dir
                visualizer._ensure_output_directory()
            else:
                print("Invalid directory path.")

        elif choice == '4':
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()