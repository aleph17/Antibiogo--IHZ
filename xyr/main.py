from dataloader import orig_train_batches, vald_batches, single_batch
from xyr_model import model
from callback import DisplayCallback # ,EarlyStopping_callback, savemodel_callback, checkpoint_filepath
import wandb
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import optimAdam, EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS
import tensorflow as tf


if __name__=="__main__":
    EPOCHS = 100
    # Launch an experiment
    wandb.init(
        project="detector",
        name= f"new_loss|{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    # Add WandbMetricsLogger to log metrics
    wandb_callbacks =WandbMetricsLogger()

    model_history = model.fit(orig_train_batches,
                              epochs= config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[DisplayCallback(),wandb_callbacks]  # , +EarlyStopping_callback
                              )
    # Mark the run as finished
    model.save(f'/mloscratch/sayfiddi/hybrid/SavedModels/modelV3Small.h5')
    # wandb.finish()
