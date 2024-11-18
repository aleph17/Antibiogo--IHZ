from dataloader import orig_train_batches, vald_batches, single_batch
from xyr_model import model
from callback import DisplayCallback, savemodel_callback  # ,EarlyStopping_callback, savemodel_callback
import wandb
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import optimAdam, EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS
import tensorflow as tf

# filters = numbers of filters in the first block of upstack. must be devisable by 8

if __name__=="__main__":
    EPOCHS = 100
    # Launch an experiment
    wandb.init(
        project="detector",
        name= f"{date.today()}|test",
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
                              callbacks=[wandb_callbacks,savemodel_callback,DisplayCallback()]  #+EarlyStopping_callback
                              )
    # Mark the run as finished
    wandb.finish()
