# Load necessary libraries
import os
import numpy as np
import paddle
from rdkit import Chem
from rdkit.Chem import AllChem
from pahelix.utils.protein_tools import ProteinTokenizer
from pahelix.utils.compound_tools import mol_to_graph_data
from src.model import DTAModel, DTAModelCriterion
from src.data_gen import DTADataset, DTACollateFunc
from src.utils import concordance_index

# Set up data directories
data_dir = 'data'
davis_dir = os.path.join(data_dir, 'davis')
kiba_dir = os.path.join(data_dir, 'kiba')

# Prepare dataset
train_data_path = os.path.join(davis_dir, 'processed', 'train')
test_data_path = os.path.join(davis_dir, 'processed', 'test')
max_protein_len = 1000  # Use -1 to indicate full sequence

# Load datasets
train_dataset = DTADataset(train_data_path, max_protein_len=max_protein_len)
test_dataset = DTADataset(test_data_path, max_protein_len=max_protein_len)
print('Training set size:', len(train_dataset))
print('Testing set size:', len(test_dataset))

# Model configuration
lr = 0.0005
model_config = {
    "compound": {
        "atom_types": ["atomic_number", "chiral_tag"],
        "bond_types": ["bond_direction", "bond_type"],
        "gnn_type": "gin",
        "dropout_rate": 0.2,
        "embedding_dim": 32,
        "num_layers": 5,
        "hidden_size": 32,
        "output_dim": 128
    },
    "protein": {
        "max_protein_len": max_protein_len,
        "embedding_dim": 128,
        "num_filters": 32,
        "output_dim": 128
    },
    "dropout_rate": 0.2
}

# Initialize model, loss function, and optimizer
model = DTAModel(model_config)
criterion = DTAModelCriterion()
optimizer = paddle.optimizer.Adam(
    learning_rate=lr,
    parameters=model.parameters()
)

# Training and evaluation settings
max_epoch = 2
batch_size = 512
num_workers = 4

# Training function
def train(model, criterion, optimizer, dataloader):
    model.train()
    losses = []
    for graphs, proteins_token, proteins_mask, labels in dataloader:
        graphs = graphs.tensor()
        proteins_token = paddle.to_tensor(proteins_token)
        proteins_mask = paddle.to_tensor(proteins_mask)
        labels = paddle.to_tensor(labels)

        preds = model(graphs, proteins_token, proteins_mask)
        loss = criterion(preds, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        losses.append(loss.numpy())
    return np.mean(losses)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    for graphs, proteins_token, proteins_mask, labels in dataloader:
        graphs = graphs.tensor()
        proteins_token = paddle.to_tensor(proteins_token)
        proteins_mask = paddle.to_tensor(proteins_mask)

        preds = model(graphs, proteins_token, proteins_mask)
        all_preds.append(preds.numpy())
        all_labels.append(labels)

    all_preds = np.concatenate(all_preds, 0).flatten()
    all_labels = np.concatenate(all_labels, 0).flatten()
    mse = ((all_labels - all_preds) ** 2).mean(axis=0)

    ci = concordance_index(all_labels, all_preds)
    return mse, ci

# Training loop
best_mse, best_ci, best_epoch = np.inf, 0, 0
best_model_path = 'best_model.pdparams'

for epoch in range(1, max_epoch + 1):
    print('========== Epoch {} =========='.format(epoch))
    train_loss = train(model, criterion, optimizer, train_dataloader)
    print('Epoch: {}, Train Loss: {:.6f}'.format(epoch, train_loss))
    mse, ci = evaluate(model, test_dataloader)
    
    if mse < best_mse:
        best_mse, best_ci, best_epoch = mse, ci, epoch  
        paddle.save(model.state_dict(), best_model_path)
        print('Saved model with MSE {:.6f} and CI {:.6f} at epoch {}'.format(best_mse, best_ci, best_epoch))
    else:
        print('No improvement in MSE.')

# Inference
# Define example drug and protein
protein_sequence = 'MENKKKDKDKSDDRMARPSGRSGHNTRGTGSSSSGVLMVGPNFRVGKKIGCGNFGELRLGKNLYTNEYVAIKLEPMKSRAPQLHLEYRFYKQLGSGDGIPQVYYFGPCGKYNAMVLELLGPSLEDLFDLCDRTFSLKTVLMIAIQLISRMEYVHSKNLIYRDVKPENFLIGRPGNKTQQVIHIIDFGLAKEYIDPETKKHIPYREHKSLTGTARYMSINTHLGKEQSRRDDLEALGHMFMYFLRGSLPWQGLKADTLKERYQKIGDTKRATPIEVLCENFPEMATYLRYVRRLDFFEKPDYDYLRKLFTDLFDRKGYMFDYEYDWIGKQLPTPVGAVQQDPALSSNREAHQHRDKMQQSKNQSADHRAAWDSQQANPHHLRAHLAADRHGGSVQVVSSTNGELNTDDPTAGRSNAPITAPTEVEVMDETKCCCFFKRRKRKTIQRHK'
drug_smiles = 'CCN1C2=C(C=CC(=C2)OC)SC1=CC(=O)C'

# Process drug to graph
mol = AllChem.MolFromSmiles(drug_smiles)
mol_graph = mol_to_graph_data(mol)

# Process protein sequence
tokenizer = ProteinTokenizer()
protein_token_ids = tokenizer.gen_token_ids(protein_sequence)

# Merge drug and protein data
data = {k: v for k, v in mol_graph.items()}
data['protein_token_ids'] = np.array(protein_token_ids)

# Truncate or pad protein sequence if necessary
if max_protein_len > 0:
    protein_token_ids = np.zeros(max_protein_len, dtype=np.int64) + ProteinTokenizer.padding_token_id
    n = min(max_protein_len, data['protein_token_ids'].size)
    protein_token_ids[:n] = data['protein_token_ids'][:n]
    data['protein_token_ids'] = protein_token_ids

# Prepare for inference
join_graph, proteins_token, proteins_mask = infer_collate_fn([data])
join_graph = join_graph.tensor()
proteins_token = paddle.to_tensor(proteins_token)
proteins_mask = paddle.to_tensor(proteins_mask)

# Perform inference
model.eval()
affinity_pred = model(join_graph, proteins_token, proteins_mask).numpy()[0][0]

# Convert affinity prediction to Kd value
Kd = 10 ** (-affinity_pred)
print('Predicted Kd value:', Kd)
