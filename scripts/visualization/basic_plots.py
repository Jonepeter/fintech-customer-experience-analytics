"""
    Visualization class: contain all visualization scripts 
"""

import matplotlib.pyplot as plt
import seaborn as sns


    
def bar_plot_generator(self, data, x_col, y_col, title, xlabel, ylabel):
    sns.set_style('whitegrid')
    
    try:
        plt.figure(figsize=(10,6))
        sns.barplot(data=data, x=x_col, y=y_col, color='skyblue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()  
    except ImportError as e:
        print(f"Error importing libraries: {e}. Please ensure Seaborn and Matplotlib are installed.")
    except ValueError as e:
        print(f"Error in data processing or plotting: {e}. Check your data format and values.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def count_plot(data, x_col, title, xlabel):
    try:
        # Define a color palette for multiple colors
        palette = sns.color_palette("husl", len(data[x_col].unique()))
        # Set Seaborn style
        sns.set_style("whitegrid")

        # Create count plot
        plt.figure(figsize=(8, 6))  # Set figure size
        sns.countplot(x=x_col, data=data, palette=palette, edgecolor='black')

        # Customize the plot
        plt.title(f'{title}', fontsize=14, pad=15)
        plt.xlabel(f'{xlabel}', fontsize=12)
        plt.ylabel('Count Values', fontsize=12)
        # Display the plot
        plt.tight_layout()
        plt.show()

    except ImportError as e:
        print(f"Error importing libraries: {e}. Please ensure Seaborn, Matplotlib, and Pandas are installed.")
    except ValueError as e:
        print(f"Error in data processing or plotting: {e}. Check your data format and values.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def histogram_plot(data, x_col, title, xlabel, ylabel):
    try:
        # Set Seaborn style
        sns.set_style("whitegrid")

        # Define a color palette for multiple colors
        palette = sns.color_palette("husl", len(data[x_col].unique()))

        # Create histogram
        plt.figure(figsize=(8, 6))  # Set figure size
        for group, color in zip(data[x_col].unique(), palette):
            sns.histplot(
                data=data[data[x_col] == group],
                x='Values',
                color=color,
                label=group,
                alpha=0.6,  # Slight transparency for overlap
                edgecolor='black',
                bins=20
            )

        # Customize the plot
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(xlabel=xlabel, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title=x_col)

        # Display the plot
        plt.tight_layout()
        plt.show()

    except ImportError as e:
        print(f"Error importing libraries: {e}. Please ensure Seaborn, Matplotlib, Pandas, and NumPy are installed.")
    except ValueError as e:
        print(f"Error in data processing or plotting: {e}. Check your data format and values.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def line_plot(self):
    pass