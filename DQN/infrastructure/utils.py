from infrastructure.config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA, Transition

def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))
    