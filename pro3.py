import py3Dmol
import requests

def fetch_pdb_structure(pdb_id):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch PDB structure for {pdb_id}")
        return None

def visualize_protein_structure(pdb_data):
    view = py3Dmol.view(width=400, height=300)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view.show()

def main():
    pdb_ids = ['1L2Y', '2NRL', '3R9A', '4ZV6', '5T4N', '6WZC', '7CWF', '8OQW','8JGC','8QQZ','8IU7','8ORC']
    for pdb_id in pdb_ids:
        pdb_data = fetch_pdb_structure(pdb_id)
        if pdb_data:
            visualize_protein_structure(pdb_data)
        else:
            print(f"Skipping {pdb_id} due to failure in fetching its structure.")

if __name__ == "__main__":
    main()
