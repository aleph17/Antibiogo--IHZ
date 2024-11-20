from dataloader import orig_train_batches, vald_batches, single_batch
from callback import DisplayCallback, savemodel_callback  # ,EarlyStopping_callback, savemodel_callback
import wandb
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import optimAdam, EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS
import tensorflow as tf
import sys

# filters = numbers of filters in the first block of upstack. must be devisable by 8

if __name__=="__main__":
    EPOCHS = 100
    # Launch an experiment
    choice = int(sys.argv[1])
    option = {0:'V3', 1:'VIT'}
    if choice == 0:
        from pointer import model
    if choice == 1:
        from mobilevit import model
    wandb.init(
        project="pointer",
        name= f"pointer{option[choice]}'|{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    # # Add WandbMetricsLogger to log metrics
    wandb_callbacks =WandbMetricsLogger()
    model_history = model.fit(orig_train_batches,
                              epochs=config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[wandb_callbacks,DisplayCallback()]  #+EarlyStopping_callback, savemodel_callback
                              )
    model.save(f'pointer/SavedModels/mobile{option[choice]}.h5')
    # Mark the run as finished
    wandb.finish()
