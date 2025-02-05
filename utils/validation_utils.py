class ValidationMonitor:
    '''
    Monitors the validation loss and iou to check if the model is improving.
    '''
    def __init__(self, patience: int=1, delta: float=0.) -> None:
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_mean_iou = 0.
        self.min_val_loss = float('inf')
    
    def improved_model(self, mean_iou: float) -> bool:
        if mean_iou > self.best_mean_iou:
            self.best_mean_iou = mean_iou
            return True

        return False

    def reached_plateau(self, val_loss: float) -> bool:
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0

        elif val_loss > (self.min_val_loss + self.delta):
            self.counter += 1

            if self.counter >= self.patience:
                return True

        return False