!pip install py3Dmol
from IPython.display import display

import py3Dmol

def visualize_protein_structure(pdb_data):
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view.show()

def main():

    pdb_data = """\
ATOM      1  N   MET A   1       2.360  39.301  40.057  1.00 17.00           N
ATOM      2  CA  MET A   1       1.816  39.880  41.305  1.00 17.00           C
ATOM      3  C   MET A   1       1.648  41.391  41.178  1.00 17.00           C
ATOM      4  O   MET A   1       1.951  42.149  42.078  1.00 17.00           O
ATOM      5  CB  MET A   1       0.465  39.308  41.733  1.00 17.00           C
ATOM      6  CG  MET A   1      -0.293  39.819  42.975  1.00 17.00           C
ATOM      7  SD  MET A   1      -1.764  39.268  43.272  1.00 17.00           S
ATOM      8  CE  MET A   1      -2.288  40.605  44.372  1.00 17.00           C
ATOM      9  N   VAL A   2       1.105  41.812  40.011  1.00 16.92           N
ATOM     10  CA  VAL A   2       0.881  43.236  39.889  1.00 16.92           C
ATOM     11  C   VAL A   2       2.071  44.106  40.342  1.00 16.92           C
ATOM     12  O   VAL A   2       1.961  44.860  41.287  1.00 16.92           O
ATOM     13  CB  VAL A   2      -0.409  43.472  39.451  1.00 16.92           C
ATOM     14  CG1 VAL A   2      -0.631  44.924  39.067  1.00 16.92           C
ATOM     15  CG2 VAL A   2      -1.485  42.570  39.874  1.00 16.92           C
ATOM     16  N   ARG A   3       3.223  43.966  39.729  1.00 16.37           N
ATOM     17  CA  ARG A   3       4.421  44.706  40.108  1.00 16.37           C
ATOM     18  C   ARG A   3       4.349  46.188  40.063  1.00 16.37           C
ATOM     19  O   ARG A   3       4.298  46.794  41.119  1.00 16.37           O
ATOM     20  CB  ARG A   3       5.758  43.966  39.782  1.00 16.37           C
ATOM     21  CG  ARG A   3       6.895  44.783  40.357  1.00 16.37           C
ATOM     22  CD  ARG A   3       8.149  43.991  40.102  1.00 16.37           C
ATOM     23  NE  ARG A   3       8.222   42.651  40.724  1.00 16.37           N
ATOM     24  CZ  ARG A   3       9.351   41.902  40.678  1.00 16.37           C
ATOM     25  NH1 ARG A   3      10.618   42.345  40.344  1.00 16.37           N
ATOM     26  NH2 ARG A   3       9.247   40.651  41.002  1.00 16.37           N
ATOM     27  N   THR A   4       4.314  46.760  38.825  1.00 16.00           N
ATOM     28  CA  THR A   4       4.346  48.207  38.686  1.00 16.00           C
ATOM     29  C   THR A   4       5.749  48.635  38.250  1.00 16.00           C
ATOM     30  O   THR A   4       6.290  49.634  38.745  1.00 16.00           O
ATOM     31  CB  THR A   4       3.646  48.712  37.396  1.00 16.00           C
ATOM     32  OG1 THR A   4       2.301  48.214  37.532  1.00 16.00           O
ATOM     33  CG2 THR A   4       4.425  50.163  37.398"""


    display(visualize_protein_structure(pdb_data))

if __name__ == "__main__":
    main()
