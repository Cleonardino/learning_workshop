import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import import_datasets, prepare_data, prepare_label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from tqdm import tqdm
import time

# -------------------------------------------------------------------
# 1. ARCHITECTURE TRANSFORMER
# -------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, output_dim=4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
    def forward(self, src, src_mask=None):
        # src shape: (batch, seq_len, input_dim)
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        # Prendre la sortie de la dernière position temporelle
        x = x[:, -1, :]
        output = self.output_projection(x)
        return output


# -------------------------------------------------------------------
# 2. DATASET PYTORCH POUR SÉQUENCES TEMPORELLES
# -------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=24):
        """
        X: DataFrame ou array des features
        y: DataFrame ou array des labels
        sequence_length: nombre de pas de temps à considérer
        """
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.DataFrame) else y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Séquence d'entrée: de idx à idx+sequence_length
        X_seq = self.X[idx:idx + self.sequence_length]
        # Target: valeur au dernier pas de temps
        y_target = self.y[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)


# -------------------------------------------------------------------
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# -------------------------------------------------------------------

train_data, train_labels, test_data = import_datasets()

train_data = prepare_data(train_data)
train_labels = prepare_label(train_labels)
test_data = prepare_data(test_data)

X_cols = [
    "minutes_since_Epoch",
    "consumption",
    "visibility",
    "temperature",
    "humidity",
    "humidex",
    "windchill",
    "wind",
    "pressure",
    "dayofweek",
    "isweekend",
    "saison",
    "ispublicholiday",
    "isbusinesshour"
]

y_cols = [
    "washing_machine",
    "fridge_freezer",
    "TV",
    "kettle"
]

X = train_data[X_cols]
y = train_labels[y_cols]

# -------------------------------------------------------------------
# 4. MASQUE DE VALIDITÉ (Y sans NaN)
# -------------------------------------------------------------------

valid_y_mask = ~y.isna().any(axis=1)
X_valid = X.loc[valid_y_mask]
y_valid = y.loc[valid_y_mask]

# -------------------------------------------------------------------
# 5. SPLIT TEMPOREL (sans shuffle)
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, shuffle=False
)

# -------------------------------------------------------------------
# 6. NORMALISATION (X et Y)
# -------------------------------------------------------------------

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Recréer des DataFrames pour faciliter l'indexation
X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_cols)
X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_cols)
y_train_df = pd.DataFrame(y_train_scaled, index=y_train.index, columns=y_cols)
y_test_df = pd.DataFrame(y_test_scaled, index=y_test.index, columns=y_cols)

# -------------------------------------------------------------------
# 7. CRÉATION DES DATASETS ET DATALOADERS
# -------------------------------------------------------------------

SEQUENCE_LENGTH = 48  # Augmenté pour plus de contexte
BATCH_SIZE = 128  # Augmenté pour stabilité

train_dataset = TimeSeriesDataset(X_train_df, y_train_df, sequence_length=SEQUENCE_LENGTH)
test_dataset = TimeSeriesDataset(X_test_df, y_test_df, sequence_length=SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Shuffle activé
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------------------
# 8. INITIALISATION DU MODÈLE
# -------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de: {device}")

model = TimeSeriesTransformer(
    input_dim=len(X_cols),
    d_model=64,  # Réduit pour éviter overfitting
    nhead=4,  # Réduit
    num_layers=3,  # Réduit
    dim_feedforward=256,  # Réduit
    dropout=0.2,  # Augmenté pour régularisation
    output_dim=len(y_cols)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)  # AdamW avec weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# -------------------------------------------------------------------
# 9. ENTRAÎNEMENT AVEC MÉTRIQUES DÉTAILLÉES
# -------------------------------------------------------------------

def compute_metrics(y_true, y_pred, scaler_y):
    """Calcule plusieurs métriques de qualité - dénormalise d'abord"""
    # Dénormaliser les prédictions et vraies valeurs
    y_true_original = scaler_y.inverse_transform(y_true)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    metrics = {}
    for i, col in enumerate(y_cols):
        mae = mean_absolute_error(y_true_original[:, i], y_pred_original[:, i])
        mse = mean_squared_error(y_true_original[:, i], y_pred_original[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
        
        # MAPE (Mean Absolute Percentage Error) - éviter division par zéro
        mask = np.abs(y_true_original[:, i]) > 1e-6  # Éviter division par des valeurs très petites
        if mask.any():
            mape = np.mean(np.abs((y_true_original[mask, i] - y_pred_original[mask, i]) / y_true_original[mask, i])) * 100
        else:
            mape = 0
        
        metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': min(mape, 999.99)  # Limiter MAPE pour affichage
        }
    return metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader), np.vstack(all_preds), np.vstack(all_targets)


# Entraînement avec suivi détaillé
NUM_EPOCHS = 50
best_val_loss = float('inf')
best_metrics = None
history = {
    'train_loss': [],
    'val_loss': [],
    'epoch_time': []
}

print("="*70)
print("DÉBUT DE L'ENTRAÎNEMENT DU TRANSFORMER")
print("="*70)
print(f"Nombre d'epochs: {NUM_EPOCHS}")
print(f"Taille des batchs: {BATCH_SIZE}")
print(f"Longueur de séquence: {SEQUENCE_LENGTH}")
print(f"Device: {device}")
print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
print("="*70)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Entraînement
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validation
    val_loss, val_preds, val_targets = evaluate(model, test_loader, criterion, device)
    
    # Calcul des métriques (avec dénormalisation)
    metrics = compute_metrics(val_targets, val_preds, scaler_y)
    
    # Scheduler
    scheduler.step(val_loss)
    
    # Historique
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    epoch_time = time.time() - epoch_start
    history['epoch_time'].append(epoch_time)
    
    # Sauvegarde du meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_metrics = metrics
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics
        }, 'best_transformer_model.pth')
    
    # Affichage détaillé
    progress = (epoch + 1) / NUM_EPOCHS * 100
    elapsed = time.time() - start_time
    eta = elapsed / (epoch + 1) * (NUM_EPOCHS - epoch - 1)
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({progress:.1f}%) | Temps: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")
    print(f"{'='*70}")
    print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"\nMétriques de validation:")
    print(f"{'Appareil':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
    print(f"{'-'*70}")
    for col, m in metrics.items():
        print(f"{col:<20} {m['MAE']:<10.3f} {m['RMSE']:<10.3f} {m['R2']:<10.3f} {m['MAPE']:<10.2f}%")
    
    # Moyenne des métriques
    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    print(f"{'-'*70}")
    print(f"{'MOYENNE':<20} {avg_mae:<10.3f} {'':<10} {avg_r2:<10.3f}")
    
    if val_loss == best_val_loss:
        print(f"\n🌟 Meilleur modèle sauvegardé!")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"ENTRAÎNEMENT TERMINÉ EN {total_time/60:.1f} minutes")
print(f"{'='*70}")

# -------------------------------------------------------------------
# 10. ÉVALUATION FINALE AVEC MÉTRIQUES COMPLÈTES
# -------------------------------------------------------------------

print("\n" + "="*70)
print("ÉVALUATION FINALE SUR LE MEILLEUR MODÈLE")
print("="*70)

# Charger le meilleur modèle
checkpoint = torch.load('best_transformer_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

_, final_preds, final_targets = evaluate(model, test_loader, criterion, device)
final_metrics = compute_metrics(final_targets, final_preds, scaler_y)

print(f"\nMeilleur modèle obtenu à l'epoch {checkpoint['epoch'] + 1}")
print(f"Validation Loss: {checkpoint['val_loss']:.6f}\n")

print(f"{'Appareil':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
print(f"{'='*70}")
for col, m in final_metrics.items():
    print(f"{col:<20} {m['MAE']:<10.3f} {m['RMSE']:<10.3f} {m['R2']:<10.3f} {m['MAPE']:<10.2f}%")

avg_mae = np.mean([m['MAE'] for m in final_metrics.values()])
avg_rmse = np.mean([m['RMSE'] for m in final_metrics.values()])
avg_r2 = np.mean([m['R2'] for m in final_metrics.values()])
avg_mape = np.mean([m['MAPE'] for m in final_metrics.values()])

print(f"{'='*70}")
print(f"{'MOYENNE':<20} {avg_mae:<10.3f} {avg_rmse:<10.3f} {avg_r2:<10.3f} {avg_mape:<10.2f}%")
print(f"{'='*70}")

# Visualisation de l'historique
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Courbes de Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning curve avec log scale
axes[1].semilogy(history['train_loss'], label='Train Loss', linewidth=2)
axes[1].semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('Courbes de Loss (échelle log)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\nGraphique de l'historique sauvegardé dans 'training_history.png'")

# -------------------------------------------------------------------
# 11. FONCTION DE PRÉDICTION AVEC GESTION DES NaN
# -------------------------------------------------------------------

def predict_with_transformer(model, scaler_X, scaler_y, X, y_cols, sequence_length=48, device='cpu'):
    """
    Prédit avec le Transformer sur des séquences.
    Retourne NaN pour les lignes avec NaN ou séquences incomplètes.
    """
    model.eval()
    predictions = pd.DataFrame(index=X.index, columns=y_cols, dtype=float)
    
    X_scaled = scaler_X.transform(X)
    
    with torch.no_grad():
        for i in range(sequence_length - 1, len(X)):
            # Vérifier si la séquence contient des NaN
            sequence = X_scaled[i - sequence_length + 1:i + 1]
            if np.isnan(sequence).any():
                continue
            
            # Prédire
            X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            pred_scaled = model(X_tensor).cpu().numpy()[0]
            
            # Dénormaliser la prédiction
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            predictions.iloc[i] = pred
    
    return predictions


# -------------------------------------------------------------------
# 12. PRÉDICTIONS SUR LE JEU DE TEST
# -------------------------------------------------------------------

test_predictions = predict_with_transformer(
    model,
    scaler_X,
    scaler_y,
    test_data[X_cols],
    y_cols,
    sequence_length=SEQUENCE_LENGTH,
    device=device
)

print("\n=== PRÉDICTIONS SUR TEST ===")
print(test_predictions.head(30))

# -------------------------------------------------------------------
# 13. RÉSUMÉ FINAL
# -------------------------------------------------------------------

print("\n" + "="*70)
print("RÉSUMÉ DE L'ENTRAÎNEMENT")
print("="*70)
print(f"Durée totale: {total_time/60:.1f} minutes")
print(f"Temps moyen par epoch: {np.mean(history['epoch_time']):.1f}s")
print(f"Meilleur epoch: {checkpoint['epoch'] + 1}")
print(f"Meilleure validation loss: {best_val_loss:.6f}")
print(f"Loss finale: {history['val_loss'][-1]:.6f}")
print(f"Amélioration: {(history['val_loss'][0] - best_val_loss) / history['val_loss'][0] * 100:.1f}%")
print("="*70)

# -------------------------------------------------------------------
# 14. SAUVEGARDE DU MODÈLE FINAL
# -------------------------------------------------------------------

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'X_cols': X_cols,
    'y_cols': y_cols,
    'sequence_length': SEQUENCE_LENGTH,
    'model_config': {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.2
    }
}, 'transformer_model.pth')

print("\nModèle sauvegardé dans 'transformer_model.pth'")