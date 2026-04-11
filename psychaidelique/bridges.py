"""
Psyk-AI-deliK - Cross-Layer Bridges
Implementing learned gated projections between distal transformer layers.
"""

import mlx.core as mx
import mlx.nn as nn

class CrossLayerBridge(nn.Module):
    """
    Crée un 'pont' entre une couche source (bas niveau) et une couche cible (haut niveau).
    C'est l'analogue technique de la connectivité accrue entre réseaux normalement ségrégués.
    """
    def __init__(self, dims: int, dropout: float = 0.1):
        super().__init__()
        # Une projection linéaire pour aligner les espaces sémantiques si nécessaire
        self.projection = nn.Linear(dims, dims)
        self.gate = nn.Linear(dims, 1) # Un mécanisme de porte pour contrôler le flux
        self.dropout = nn.Dropout(dropout)

    def __call__(self, source_hidden_states: mx.array, target_hidden_states: mx.array, intensity: float = 0.5):
        """
        Injecte l'information de la couche source dans la couche cible.
        intensity: règle la 'perméabilité' du pont (0.0 = fermé, 1.0 = ouvert).
        """
        # Transformation de la source
        projected_source = self.projection(source_hidden_states)
        
        # Calcul du gating (quelles informations de la source sont pertinentes pour la cible)
        gate_value = mx.sigmoid(self.gate(projected_source)) * intensity
        
        # Fusion par addition résiduelle pondérée
        # On court-circuite la hiérarchie standard
        bridged_output = target_hidden_states + (gate_value * self.dropout(projected_source))
        
        return bridged_output

class BridgeManager:
    """
    Gère l'ensemble des ponts entre les couches unimodales (10-30%) 
    et transmodales (60-80%).
    """
    def __init__(self, num_layers: int, model_dims: int):
        self.bridges = {}
        # Définition des zones de couplage selon la méga-analyse BOLD
        self.source_range = range(int(num_layers * 0.1), int(num_layers * 0.3))
        self.target_range = range(int(num_layers * 0.6), int(num_layers * 0.8))

    def get_bridge(self, source_idx: int, target_idx: int, dims: int):
        key = f"{source_idx}_{target_idx}"
        if key not in self.bridges:
            self.bridges[key] = CrossLayerBridge(dims)
        return self.bridges[key]
