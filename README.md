# Chess RL

Proyecto de Reinforcement Learning para ajedrez con DQN.

## Objetivo

Comprender y construir todos los componentes de una IA de ajedrez basada en DQN.

## Filosofía

- Desarrollo incremental
- Tests primero
- Arquitectura antes que velocidad
- Decisiones documentadas

## Estado actual

Actualmente el proyecto implementa un pipeline completo de entrenamiento DQN contra un oponente aleatorio:

- Entorno de ajedrez basado en python-chess.
- Codificación del tablero mediante tensores.
- CNN para estimación de Q-values.
- Replay Buffer con muestreo aleatorio.
- Entrenamiento mediante mini-batches.
- Entrenamiento durante múltiples episodios.
- Sincronización periódica de la target network.
- Reducción progresiva de epsilon tras cada actualización real de entrenamiento.

El siguiente objetivo es incorporar métricas de entrenamiento para evaluar la evolución del aprendizaje del agente.