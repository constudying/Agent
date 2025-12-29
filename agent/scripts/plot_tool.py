import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union, Tuple
import os

class PlotTool:
    """
    A plotting class for analyzing data generated during training loops.
    Supports both online (real-time) and offline plotting modes.
    Currently supports line plots and attention heatmaps.
    """

    def __init__(self, mode: str = 'offline', figsize: Tuple[int, int] = (10, 6), offscreen_rendering: bool = True):
        """
        Initialize the PlotTool.

        Args:
            mode: 'online' for real-time updating during training, 'offline' for post-training plotting
            figsize: Default figure size for plots
            offscreen_rendering: If True, use non-interactive backend (no windows pop up).
                               If False, use interactive backend (windows may pop up)
        """
        self.mode = mode
        self.figsize = figsize
        self.offscreen_rendering = offscreen_rendering
        self.figures = []
        self.axes = []
        self.current_figure = None
        self.current_axes = None

        # Online subplot grid attributes
        self.online_subplot_grid = None
        self.online_subplot_axes = None
        self.online_subplot_data = []  # List of dicts for each subplot
        self.online_subplot_config = {}  # Configuration for each subplot

        # Set matplotlib backend based on offscreen_rendering setting
        if not self.offscreen_rendering:
            # Switch to interactive backend if offscreen_rendering is False
            plt.switch_backend('TkAgg')  # or other interactive backend as available
            if self.mode == 'online':
                plt.ion()  # Enable interactive mode for real-time updates
        else:
            # Ensure we're using non-interactive backend
            if self.mode == 'online':
                # For online mode with offscreen rendering, we still need some interactivity
                # but we'll handle it carefully to avoid popping windows
                pass

    def set_offscreen_rendering(self, offscreen: bool):
        """
        Set whether to use offscreen rendering (no windows pop up).
        
        Args:
            offscreen: If True, use non-interactive backend. If False, use interactive backend.
        """
        if offscreen != self.offscreen_rendering:
            self.offscreen_rendering = offscreen
            if offscreen:
                # Switch to non-interactive backend
                plt.switch_backend('Agg')
            else:
                # Switch to interactive backend
                plt.switch_backend('TkAgg')
                if self.mode == 'online':
                    plt.ion()

    def create_figure(self, title: str = None, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure and axes.

        Args:
            title: Title for the figure
            figsize: Size of the figure

        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)

        self.figures.append(fig)
        self.axes.append(ax)
        self.current_figure = fig
        self.current_axes = ax

        return fig, ax

    def plot_line_online(self, epoch: int, y_data: Union[float, List[float]],
                        labels: Optional[List[str]] = None, title: str = None,
                        xlabel: str = 'Epoch', ylabel: str = 'Value') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot line chart in online mode. Updates the plot with new data for each epoch.

        Args:
            epoch: Current epoch number
            y_data: Y-axis data (single value or list for multiple lines)
            labels: Labels for each line (if y_data is a list)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            Tuple of (figure, axes)
        """
        if not isinstance(y_data, list):
            y_data = [y_data]
            if labels is None:
                labels = ['Value']

        if labels is None:
            labels = [f'Line {i+1}' for i in range(len(y_data))]

        if self.current_figure is None or len(self.current_axes.lines) == 0:
            # First call, create new plot
            fig, ax = self.create_figure(title, self.figsize)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Initialize lines
            self.lines = []
            self.x_data = []
            self.y_data_list = []

            for i, label in enumerate(labels):
                line, = ax.plot([], [], label=label)
                self.lines.append(line)
                self.x_data.append([])
                self.y_data_list.append([])
        else:
            ax = self.current_axes

        # Update data
        for i, y_val in enumerate(y_data):
            self.x_data[i].append(epoch)
            self.y_data_list[i].append(y_val)

            self.lines[i].set_data(self.x_data[i], self.y_data_list[i])

        # Update axis limits
        all_x = [x for sublist in self.x_data for x in sublist]
        all_y = [y for sublist in self.y_data_list for y in sublist]

        if all_x:
            ax.set_xlim(min(all_x), max(all_x))
        if all_y:
            ax.set_ylim(min(all_y) * 0.9, max(all_y) * 1.1)

        ax.legend()
        plt.draw()
        
        # Only pause if not using offscreen rendering
        if not self.offscreen_rendering:
            plt.pause(0.01)  # Small pause to allow update

        return self.current_figure, self.current_axes

    def plot_line_offline(self, epochs: List[int], y_data_list: List[List[float]],
                         labels: Optional[List[str]] = None, title: str = None,
                         xlabel: str = 'Epoch', ylabel: str = 'Value') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot line chart in offline mode using all epoch data at once.

        Args:
            epochs: List of epoch numbers
            y_data_list: List of y-data lists (one per line)
            labels: Labels for each line
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            Tuple of (figure, axes)
        """
        if labels is None:
            labels = [f'Line {i+1}' for i in range(len(y_data_list))]

        fig, ax = self.create_figure(title, self.figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for i, y_data in enumerate(y_data_list):
            ax.plot(epochs, y_data, label=labels[i])

        ax.legend()
        plt.tight_layout()

        return fig, ax

    def plot_heatmap(self, data: np.ndarray, title: str = None,
                    xlabel: str = None, ylabel: str = None,
                    cmap: str = 'viridis') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot attention heatmap.

        Args:
            data: 2D array for heatmap
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap for heatmap

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = self.create_figure(title, self.figsize)

        im = ax.imshow(data, cmap=cmap, aspect='auto')

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        plt.tight_layout()

        return fig, ax

    def create_subplot_grid(self, plots: List[Tuple[plt.Figure, plt.Axes]],
                           grid_shape: Tuple[int, int] = None,
                           titles: Optional[List[str]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Combine multiple plots into a single figure with subplots.

        Args:
            plots: List of (figure, axes) tuples to combine
            grid_shape: Shape of subplot grid (rows, cols). If None, auto-calculate.
            titles: Titles for each subplot

        Returns:
            Tuple of (combined figure, axes array)
        """
        n_plots = len(plots)

        if grid_shape is None:
            # Auto-calculate grid shape
            rows = int(np.ceil(np.sqrt(n_plots)))
            cols = int(np.ceil(n_plots / rows))
            grid_shape = (rows, cols)

        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(self.figsize[0] * grid_shape[1], self.figsize[1] * grid_shape[0]))

        if grid_shape[0] == 1 and grid_shape[1] == 1:
            axes = np.array([[axes]])
        elif grid_shape[0] == 1:
            axes = axes.reshape(1, -1)
        elif grid_shape[1] == 1:
            axes = axes.reshape(-1, 1)

        for i, (orig_fig, orig_ax) in enumerate(plots):
            if i < grid_shape[0] * grid_shape[1]:
                row = i // grid_shape[1]
                col = i % grid_shape[1]

                # Copy the content to the subplot
                for line in orig_ax.get_lines():
                    axes[row, col].plot(line.get_xdata(), line.get_ydata(), label=line.get_label())

                # Copy heatmap if present
                images = orig_ax.get_images()
                if images:
                    axes[row, col].imshow(images[0].get_array(), cmap=images[0].get_cmap(), aspect='auto')

                axes[row, col].set_title(titles[i] if titles and i < len(titles) else f'Plot {i+1}')
                axes[row, col].set_xlabel(orig_ax.get_xlabel())
                axes[row, col].set_ylabel(orig_ax.get_ylabel())

                if orig_ax.get_legend():
                    axes[row, col].legend()

        # Hide unused subplots
        for i in range(n_plots, grid_shape[0] * grid_shape[1]):
            row = i // grid_shape[1]
            col = i % grid_shape[1]
            axes[row, col].set_visible(False)

        plt.tight_layout()

        self.figures.append(fig)
        self.axes.append(axes)

        return fig, axes

    def show_figure(self, figure: plt.Figure = None):
        """
        Display the figure.

        Args:
            figure: Specific figure to show. If None, shows current figure.
        """
        if self.offscreen_rendering:
            # Skip showing figure when offscreen rendering is enabled
            print("Offscreen rendering enabled - figure not displayed. Use save_figure() to save plots.")
            return

        if figure is None:
            figure = self.current_figure

        if figure:
            if self.mode == 'online':
                plt.show(block=False)
            else:
                plt.show()

    def save_figure(self, filename: str, figure: plt.Figure = None, dpi: int = 300):
        """
        Save the figure to file.

        Args:
            filename: Path to save the figure
            figure: Specific figure to save. If None, saves current figure.
            dpi: Resolution for saved image
        """
        if figure is None:
            figure = self.current_figure

        if figure:
            # Ensure directory exists
            dir_path = os.path.dirname(filename)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # For online mode, ensure the figure is properly rendered before saving
            if self.mode == 'online':
                figure.canvas.draw()
                figure.canvas.flush_events()

            figure.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {filename}")

    def close_all(self):
        """
        Close all figures.
        """
        plt.close('all')
        self.figures = []
        self.axes = []
        self.current_figure = None
        self.current_axes = None

        # Reset online subplot grid
        self.online_subplot_grid = None
        self.online_subplot_axes = None
        self.online_subplot_data = []
        self.online_subplot_config = {}

    def create_online_subplot_grid(self, subplot_configs: List[dict],
                                  grid_shape: Tuple[int, int] = None,
                                  figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a grid of subplots that can be updated online simultaneously.

        Args:
            subplot_configs: List of dictionaries, each containing configuration for a subplot.
                            Each dict should have keys like:
                            - 'plot_type': 'line' or 'heatmap'
                            - 'title': subplot title
                            - 'xlabel': x-axis label
                            - 'ylabel': y-axis label
                            - 'labels': list of labels for line plots
                            - 'cmap': colormap for heatmaps
            grid_shape: Shape of subplot grid (rows, cols). If None, auto-calculate.
            figsize: Size of the overall figure. If None, auto-calculate based on grid.

        Returns:
            Tuple of (figure, axes_array)
        """
        n_subplots = len(subplot_configs)

        if grid_shape is None:
            # Auto-calculate grid shape
            rows = int(np.ceil(np.sqrt(n_subplots)))
            cols = int(np.ceil(n_subplots / rows))
            grid_shape = (rows, cols)

        if figsize is None:
            figsize = (self.figsize[0] * grid_shape[1], self.figsize[1] * grid_shape[0])

        # Create the main figure with subplots
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)

        # Handle different grid shapes
        if grid_shape[0] == 1 and grid_shape[1] == 1:
            axes = np.array([[axes]])
        elif grid_shape[0] == 1:
            axes = axes.reshape(1, -1)
        elif grid_shape[1] == 1:
            axes = axes.reshape(-1, 1)

        # Initialize data structures for each subplot
        self.online_subplot_data = []
        self.online_subplot_config = {}

        for i, config in enumerate(subplot_configs):
            if i < grid_shape[0] * grid_shape[1]:
                row = i // grid_shape[1]
                col = i % grid_shape[1]
                ax = axes[row, col]

                # Set subplot title and labels
                ax.set_title(config.get('title', f'Subplot {i+1}'))
                ax.set_xlabel(config.get('xlabel', ''))
                ax.set_ylabel(config.get('ylabel', ''))

                # Initialize data based on plot type
                plot_type = config.get('plot_type', 'line')
                subplot_data = {
                    'plot_type': plot_type,
                    'epoch': 0,
                    'x_data': [],
                    'y_data_list': [],
                    'lines': [],
                    'config': config
                }

                if plot_type == 'line':
                    labels = config.get('labels', [f'Line {j+1}' for j in range(len(config.get('initial_data', [[]])))])

                    # Initialize empty lines
                    for j, label in enumerate(labels):
                        line, = ax.plot([], [], label=label)
                        subplot_data['lines'].append(line)
                        subplot_data['x_data'].append([])
                        subplot_data['y_data_list'].append([])

                    ax.legend()

                elif plot_type == 'heatmap':
                    # For heatmaps, we'll update the image data
                    subplot_data['image'] = None
                    subplot_data['cmap'] = config.get('cmap', 'viridis')

                self.online_subplot_data.append(subplot_data)
                self.online_subplot_config[i] = config

        # Hide unused subplots
        for i in range(n_subplots, grid_shape[0] * grid_shape[1]):
            row = i // grid_shape[1]
            col = i % grid_shape[1]
            axes[row, col].set_visible(False)

        plt.tight_layout()

        # Store references
        self.online_subplot_grid = fig
        self.online_subplot_axes = axes
        self.figures.append(fig)
        self.axes.append(axes)

        return fig, axes

    def update_online_subplot_grid(self, subplot_data_list: List[Union[List[float], np.ndarray]],
                                  epoch: int = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Update all subplots in the online grid with new data.

        Args:
            subplot_data_list: List of data for each subplot. For line plots, this should be
                              a list of y-values (single value or list). For heatmaps, this
                              should be a 2D array.
            epoch: Current epoch number. If None, uses internal counter.

        Returns:
            Tuple of (figure, axes_array)
        """
        if self.online_subplot_grid is None:
            raise ValueError("Online subplot grid not created. Call create_online_subplot_grid first.")

        if len(subplot_data_list) != len(self.online_subplot_data):
            raise ValueError(f"Expected {len(self.online_subplot_data)} data items, got {len(subplot_data_list)}")

        for i, (data, subplot_info) in enumerate(zip(subplot_data_list, self.online_subplot_data)):
            if i >= len(self.online_subplot_data):
                continue

            row = i // self.online_subplot_axes.shape[1]
            col = i % self.online_subplot_axes.shape[1]
            ax = self.online_subplot_axes[row, col]

            plot_type = subplot_info['plot_type']

            if plot_type == 'line':
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]

                # Update data for each line
                for j, y_val in enumerate(data):
                    if j < len(subplot_info['x_data']):
                        subplot_info['x_data'][j].append(epoch if epoch is not None else subplot_info['epoch'])
                        subplot_info['y_data_list'][j].append(y_val)

                        # Update the line
                        subplot_info['lines'][j].set_data(
                            subplot_info['x_data'][j],
                            subplot_info['y_data_list'][j]
                        )

                # Update axis limits
                all_x = [x for sublist in subplot_info['x_data'] for x in sublist]
                all_y = [y for sublist in subplot_info['y_data_list'] for y in sublist]

                if all_x:
                    ax.set_xlim(min(all_x), max(all_x))
                if all_y:
                    ax.set_ylim(min(all_y) * 0.9, max(all_y) * 1.1)

            elif plot_type == 'heatmap':
                # Update heatmap
                if subplot_info['image'] is None:
                    # First time, create the image
                    subplot_info['image'] = ax.imshow(data, cmap=subplot_info['cmap'], aspect='auto')
                    # Add colorbar
                    plt.colorbar(subplot_info['image'], ax=ax)
                else:
                    # Update existing image data
                    subplot_info['image'].set_array(data)

            subplot_info['epoch'] = epoch if epoch is not None else subplot_info['epoch'] + 1

        # Redraw the figure
        self.online_subplot_grid.canvas.draw()
        self.online_subplot_grid.canvas.flush_events()

        # Only pause if we're in interactive mode and not using offscreen rendering
        if not self.offscreen_rendering and plt.isinteractive():
            try:
                plt.pause(0.01)
            except:
                pass  # Ignore if plt.pause() fails

        return self.online_subplot_grid, self.online_subplot_axes

    def save_subplot(self, subplot_index: int, filename: str, figure: plt.Figure = None,
                    axes: np.ndarray = None, dpi: int = 300, figsize: Tuple[int, int] = None):
        """
        Save a specific subplot from a subplot grid to a separate file.

        Args:
            subplot_index: Index of the subplot to save (0-based)
            filename: Path to save the subplot image
            figure: Figure containing the subplots. If None, uses current figure
            axes: Axes array from subplot grid. If None, uses current axes
            dpi: Resolution for saved image
            figsize: Size of the new figure for the single subplot. If None, uses default figsize

        Returns:
            Tuple of (new_figure, new_axes) for the extracted subplot
        """
        # Determine which subplot grid to use
        if figure is None and axes is None:
            # Try online subplot grid first
            if self.online_subplot_grid is not None and self.online_subplot_axes is not None:
                figure = self.online_subplot_grid
                axes = self.online_subplot_axes
                is_online = True
            # Then try regular subplot grid
            elif self.current_figure is not None and self.current_axes is not None:
                figure = self.current_figure
                axes = self.current_axes
                is_online = False
            else:
                raise ValueError("No subplot grid found. Create a subplot grid first.")

        if figsize is None:
            figsize = self.figsize

        # Handle different axes shapes
        if axes.ndim == 1:
            if len(axes) == 1:
                axes = axes.reshape(1, 1)
            else:
                axes = axes.reshape(1, -1)
        elif axes.ndim == 0:
            axes = np.array([[axes]])

        # Calculate subplot position
        n_rows, n_cols = axes.shape
        if subplot_index >= n_rows * n_cols:
            raise ValueError(f"Subplot index {subplot_index} is out of range. Max index: {n_rows * n_cols - 1}")

        row = subplot_index // n_cols
        col = subplot_index % n_cols

        # Get the specific subplot
        ax = axes[row, col]

        # Create a new figure for the single subplot
        new_fig, new_ax = plt.subplots(figsize=figsize)

        # Copy the content from the original subplot
        title = ax.get_title()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        new_ax.set_title(title)
        new_ax.set_xlabel(xlabel)
        new_ax.set_ylabel(ylabel)

        # Copy line plots
        for line in ax.get_lines():
            new_ax.plot(line.get_xdata(), line.get_ydata(),
                       label=line.get_label(), color=line.get_color(),
                       linestyle=line.get_linestyle(), linewidth=line.get_linewidth())

        # Copy heatmap if present
        images = ax.get_images()
        if images:
            img = images[0]
            new_img = new_ax.imshow(img.get_array(), cmap=img.get_cmap(), aspect='auto')
            # Add colorbar to new figure
            plt.colorbar(new_img, ax=new_ax)

        # Copy legend if present
        if ax.get_legend():
            new_ax.legend()

        # Set axis limits to match original
        new_ax.set_xlim(ax.get_xlim())
        new_ax.set_ylim(ax.get_ylim())

        plt.tight_layout()

        # Save the new figure
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        new_fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Subplot {subplot_index} saved to {filename}")

        return new_fig, new_ax

    def extract_subplot(self, subplot_index: int, figure: plt.Figure = None,
                       axes: np.ndarray = None, figsize: Tuple[int, int] = None):
        """
        Extract a specific subplot from a subplot grid and return it as a new figure.

        Args:
            subplot_index: Index of the subplot to extract (0-based)
            figure: Figure containing the subplots. If None, uses current figure
            axes: Axes array from subplot grid. If None, uses current axes
            figsize: Size of the new figure. If None, uses default figsize

        Returns:
            Tuple of (new_figure, new_axes) for the extracted subplot
        """
        return self.save_subplot(subplot_index, "", figure, axes, figsize=figsize)
