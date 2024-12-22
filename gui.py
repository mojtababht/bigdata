import tkinter as tk
from tkinter import messagebox


class Gui(tk.Tk):

    def __init__(self, chart):
        super().__init__()
        self.title("Charts Viewer")
        self.chart = chart
        self.buttons = [
            ("Show Ethnicity Chart", chart.show_ethnicity_chart),
            ("Show Age Chart", chart.show_age_chart),
            ("Show Sex Chart", chart.show_sex_chart),
            ('Show Family Member with ASD Chart', chart.show_family_mem_with_asd_chart),
            ('Show Jaundice Chart', chart.show_jaundice_chart),
            ('Show PCA of A1-A10 Chart', chart.perform_pca_a1_to_a10),
            ('Show PCA: A1-A10 + Demographic Features Chart', chart.perform_pca_all),
            ('Show Feature Importance from Decision Tree Classifier A1-A10 Chart', chart.perform_feature_importance_a1_to_a10),
            ('Show Feature Importance from Decision Tree Classifier A1-A10 + Demographic Features Chart', chart.perform_feature_importance_all),
        ]
        for text, command in self.buttons:
            btn = tk.Button(self, text=text, command=command, width=80, height=2)
            btn.pack(pady=5)
        close_btn = tk.Button(self, text="Close and Continue", command=self.close_window, width=30, height=2, bg="red",
                              fg="white")
        close_btn.pack(pady=10)

    def close_window(self):
        self.destroy()


def get_categorical_columns():
    """
    Displays a Tkinter GUI to select categorical columns from the dataframe.

    Parameters:
    data (pd.DataFrame): The input dataframe containing columns to select from.

    Returns:
    list: A list of selected categorical columns.
    """
    def select_columns():
        # List to store selected columns
        selected_cols = []

        # Function to update selected columns
        def update_selection():
            selected_cols.clear()
            for i, var in enumerate(checkbox_vars):
                if var.get():
                    selected_cols.append(columns[i])

        # Function to confirm selection and close the window
        def confirm_selection():
            update_selection()
            nonlocal result
            result = selected_cols or []
            window.destroy()

        # Create a new tkinter window
        window = tk.Tk()
        window.title("Select Categorical Columns")

        # Get column names from dataset
        columns = ['Sex', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test', 'Ethnicity']

        # Create checkboxes for each column
        checkbox_vars = []
        for col in columns:
            var = tk.BooleanVar()
            checkbox_vars.append(var)
            cb = tk.Checkbutton(window, text=col, variable=var, onvalue=True, offvalue=False, font=("Arial", 12))
            cb.pack(anchor='w')

        # Add confirm button
        confirm_button = tk.Button(window, text="Confirm", command=confirm_selection, font=("Arial", 12))
        confirm_button.pack(pady=10)

        # Run the tkinter main loop
        window.mainloop()

        return result

    # Result variable to capture selected columns
    result = []
    return select_columns()
