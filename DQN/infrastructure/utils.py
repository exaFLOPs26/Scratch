from infrastructure.config import Batch_size, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return