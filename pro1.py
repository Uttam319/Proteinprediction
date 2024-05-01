from Bio.SeqUtils.ProtParam import ProteinAnalysis

def predict_secondary_structure(protein_sequence):

    analyzed_seq = ProteinAnalysis(protein_sequence)


    secondary_structure = analyzed_seq.secondary_structure_fraction()


    secondary_structure_dict = {
        'H': secondary_structure[0],  # Helix
        'E': secondary_structure[1],  # Sheet
        'C': secondary_structure[2]   # Coil
    }

    return secondary_structure_dict

def main():
    # Load protein sequence
    protein_sequence = "MKVKVTVRTLRKRKRPVRGSQKRGILTLKFLHLFLGIGQVGLILAMACFHVIGVTLPGTIYQHVRLSNLTVVTDAYFIVSLAVADYFLVTVFQTIHALRPVGRGLIGVMLSLGIILSLSLGGVVLAGTETNVVSRLLALISALAGVVVLWLQGLDRENRALVLVLRRLRSAGRRRLLLGGVLWGVAAFTIPTIGSALRLSPRGLGIGVGILVGIIYRQGLMGLLGLRRLGFRLLILGVAVRSPSLFGGGWRRRIPLTRLLRLLILVGVILAVLGLRGVLGAFTILGAFGGVFALFGKWKSPKMRRLLGHLLMLVIGFLGFYIGLVLRLLGRILTLRHIVLHPLFVTAFALWRGGLYFLGSFWLPLLHRLGLRRLVAGLPRGAVLLVLGLLISGAVGLGLRRGIPRRIGVILVVLVVVGLARSGAVRALLAAGLLLLRLLLLGFRPGPTMRLGA"

    # Predict secondary structure
    secondary_structure = predict_secondary_structure(protein_sequence)

    # Print secondary structure
    print("Secondary Structure:")
    print("Helix: {:.2f}%".format(secondary_structure['H'] * 100))
    print("Sheet: {:.2f}%".format(secondary_structure['E'] * 100))
    print("Coil: {:.2f}%".format(secondary_structure['C'] * 100))

if __name__ == "__main__":
    main()
