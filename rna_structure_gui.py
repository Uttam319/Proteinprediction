import tkinter as tk
from tkinter import ttk
import pahelix.toolkit.linear_rna as linear_rna

def predict_structure():
    sequence = sequence_entry.get()
    output_text.delete(1.0, tk.END)  # Clear previous output
    if sequence:
        output_text.insert(tk.END, "Predicted Structure:\n")
        # Predict using LinearFold
        structure, energy = linear_rna.linear_fold_c(sequence)
        output_text.insert(tk.END, f"{structure}\n")
        output_text.insert(tk.END, f"Free Energy: {energy}\n\n")
        # Predict using LinearPartition
        partition_func, bp_probabilities = linear_rna.linear_partition_c(sequence, bp_cutoff=0.2)
        output_text.insert(tk.END, f"Partition Function: {partition_func}\n")
        output_text.insert(tk.END, "Base Pair Probabilities:\n")
        for pair in bp_probabilities:
            output_text.insert(tk.END, f"{pair}\n")

root = tk.Tk()
root.title("RNA Secondary Structure Prediction")


style = ttk.Style()
style.theme_use("clam")  # Use a modern theme


input_frame = ttk.Frame(root, padding="20")
input_frame.grid(row=0, column=0, sticky="nsew")


sequence_label = ttk.Label(input_frame, text="RNA Sequence:", font=("Helvetica", 12))
sequence_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
sequence_entry = ttk.Entry(input_frame, width=40, font=("Helvetica", 10))
sequence_entry.insert(0, "GGGAAAUCCC")  # Default sequence
sequence_entry.grid(row=0, column=1, padx=5, pady=5)

predict_button = ttk.Button(input_frame, text="Predict", command=predict_structure)
predict_button.grid(row=0, column=2, padx=5, pady=5)


output_frame = ttk.Frame(root, padding="20")
output_frame.grid(row=1, column=0, sticky="nsew")


output_text = tk.Text(output_frame, wrap="word", width=60, height=20, font=("Helvetica", 10))
output_text.grid(row=0, column=0, sticky="nsew")


scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=output_text.yview)
scrollbar.grid(row=0, column=1, sticky="ns")
output_text.config(yscrollcommand=scrollbar.set)


root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
output_frame.columnconfigure(0, weight=1)
output_frame.rowconfigure(0, weight=1)

root.mainloop()
