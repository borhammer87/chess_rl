# ARCHITECTURE

## Flujo
ChessEnv -> BoardEncoder -> Tensor -> DQNCNN -> 4096 Q-values -> Legal Mask -> decode_legal_action -> ChessEnv.step -> ReplayBuffer -> train_step.

Actualizar solo cuando cambie la arquitectura.
