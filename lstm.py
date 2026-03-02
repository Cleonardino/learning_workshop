import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preparation import import_datasets, prepare_data, prepare_label, remove_nan_consumption
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. ARCHITECTURE LSTM BIDIRECTIONNEL AVEC ATTENTION
# -------------------------------------------------------------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context shape: (batch, hidden_dim)
        return context, attention_weights


class BidirectionalLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, 
                 dropout=0.3, output_dim=4):
        super().__init__()
        
        # Couche d'embedding
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM Bidirectionnel
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention sur les sorties LSTM
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 pour bidirectionnel
        
        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        context, attention_weights = self.attention(lstm_out)
        # context shape: (batch, hidden_dim * 2)
        
        # Fully connected layers
        output = self.fc_layers(context)
        
        return output


# -------------------------------------------------------------------
# 2. DATASET PYTORCH POUR SÉQUENCES TEMPORELLES
# -------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=48):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.DataFrame) else y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)


# -------------------------------------------------------------------
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# -------------------------------------------------------------------

train_data, train_labels, test_data = import_datasets()

train_data = prepare_data(train_data)
train_labels = prepare_label(train_labels)
test_data = prepare_data(test_data)
train_data = remove_nan_consumption(train_data)
test_data = remove_nan_consumption(test_data)

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

X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_cols)
X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_cols)
y_train_df = pd.DataFrame(y_train_scaled, index=y_train.index, columns=y_cols)
y_test_df = pd.DataFrame(y_test_scaled, index=y_test.index, columns=y_cols)

# -------------------------------------------------------------------
# 7. CRÉATION DES DATASETS ET DATALOADERS
# -------------------------------------------------------------------

SEQUENCE_LENGTH = 48
BATCH_SIZE = 128

train_dataset = TimeSeriesDataset(X_train_df, y_train_df, sequence_length=SEQUENCE_LENGTH)
test_dataset = TimeSeriesDataset(X_test_df, y_test_df, sequence_length=SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------------------
# 8. INITIALISATION DU MODÈLE
# -------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de: {device}")

model = BidirectionalLSTMWithAttention(
    input_dim=len(X_cols),
    hidden_dim=128,
    num_layers=3,
    dropout=0.3,
    output_dim=len(y_cols)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

# -------------------------------------------------------------------
# 9. FONCTIONS D'ENTRAÎNEMENT ET ÉVALUATION
# -------------------------------------------------------------------

def compute_metrics(y_true, y_pred, scaler_y):
    y_true_original = scaler_y.inverse_transform(y_true)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    
    metrics = {}
    for i, col in enumerate(y_cols):
        mae = mean_absolute_error(y_true_original[:, i], y_pred_original[:, i])
        mse = mean_squared_error(y_true_original[:, i], y_pred_original[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
        
        mask = np.abs(y_true_original[:, i]) > 1e-6
        if mask.any():
            mape = np.mean(np.abs((y_true_original[mask, i] - y_pred_original[mask, i]) / y_true_original[mask, i])) * 100
        else:
            mape = 0
        
        metrics[col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': min(mape, 999.99)
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
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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


# -------------------------------------------------------------------
# 10. ENTRAÎNEMENT AVEC MÉTRIQUES DÉTAILLÉES
# -------------------------------------------------------------------

NUM_EPOCHS = 100
best_val_loss = float('inf')
best_metrics = None
patience_counter = 0
EARLY_STOP_PATIENCE = 15

history = {
    'train_loss': [],
    'val_loss': [],
    'epoch_time': [],
    'learning_rates': []
}

print("="*70)
print("ENTRAÎNEMENT DU LSTM BIDIRECTIONNEL AVEC ATTENTION")
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
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_preds, val_targets = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(val_targets, val_preds, scaler_y)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rates'].append(optimizer.param_groups[0]['lr'])
    epoch_time = time.time() - epoch_start
    history['epoch_time'].append(epoch_time)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_metrics = metrics
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics
        }, 'best_lstm_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\Early stopping déclenché après {epoch + 1} epochs")
        break
    
    progress = (epoch + 1) / NUM_EPOCHS * 100
    elapsed = time.time() - start_time
    eta = elapsed / (epoch + 1) * (NUM_EPOCHS - epoch - 1)
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({progress:.1f}%) | Temps: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")
    print(f"{'='*70}")
    print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
    print(f"\nMétriques de validation:")
    print(f"{'Appareil':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
    print(f"{'-'*70}")
    for col, m in metrics.items():
        print(f"{col:<20} {m['MAE']:<10.3f} {m['RMSE']:<10.3f} {m['R2']:<10.3f} {m['MAPE']:<10.2f}%")
    
    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    print(f"{'-'*70}")
    print(f"{'MOYENNE':<20} {avg_mae:<10.3f} {'':<10} {avg_r2:<10.3f}")
    
    if val_loss == best_val_loss:
        print(f"\nMeilleur modèle sauvegardé!")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"ENTRAÎNEMENT TERMINÉ EN {total_time/60:.1f} minutes")
print(f"{'='*70}")

# -------------------------------------------------------------------
# 11. ÉVALUATION FINALE
# -------------------------------------------------------------------

print("\n" + "="*70)
print("ÉVALUATION FINALE SUR LE MEILLEUR MODÈLE")
print("="*70)

checkpoint = torch.load('best_lstm_model.pth', weights_only=False)
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

# -------------------------------------------------------------------
# 12. VISUALISATION
# -------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Courbes de Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogy(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 1].semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss (log scale)')
axes[0, 1].set_title('Courbes de Loss (échelle log)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history['learning_rates'], linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Évolution du Learning Rate')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

axes[1, 1].plot(history['epoch_time'], linewidth=2, color='orange')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Temps (secondes)')
axes[1, 1].set_title('Temps par Epoch')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé dans 'lstm_training_history.png'")

# -------------------------------------------------------------------
# 13. FONCTION DE PRÉDICTION
# -------------------------------------------------------------------

def predict_with_lstm(model, scaler_X, scaler_y, X, y_cols, sequence_length=48, device='cpu'):
    model.eval()
    predictions = pd.DataFrame(index=X.index, columns=y_cols, dtype=float)
    
    X_scaled = scaler_X.transform(X)
    
    with torch.no_grad():
        for i in range(sequence_length - 1, len(X)):
            sequence = X_scaled[i - sequence_length + 1:i + 1]
            if np.isnan(sequence).any():
                continue
            
            X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            pred_scaled = model(X_tensor).cpu().numpy()[0]
            pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            predictions.iloc[i] = pred
    
    return predictions


# -------------------------------------------------------------------
# 14. PRÉDICTIONS SUR LE JEU DE TEST
# -------------------------------------------------------------------

test_predictions = predict_with_lstm(
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
# 15. RÉSUMÉ FINAL
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
# 16. SAUVEGARDE DU MODÈLE FINAL
# -------------------------------------------------------------------

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'X_cols': X_cols,
    'y_cols': y_cols,
    'sequence_length': SEQUENCE_LENGTH,
    'model_config': {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.3
    }
}, 'lstm_model.pth')

print("\nModèle sauvegardé dans 'lstm_model.pth'")