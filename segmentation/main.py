from dataloader import orig_train_batches, vald_batches, single_batch
from basemodel import model_net_MobileNetv2, filters
from callback import DisplayCallback, checkpoint_filepath  # ,EarlyStopping_callback, savemodel_callback
import wandb
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import optimAdam, EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS
import tensorflow as tf

# filters = numbers of filters in the first block of upstack. must be devisable by 8

if __name__=="__main__":
    with open(f"ModelSummary/model_{filters}.txt", "w") as fily:
        model_net_MobileNetv2.summary(print_fn=lambda x: fily.write(x + "\n"))
    for batch in EXPR_BATCHES:
        checkpoint_filepath = f'BaseModel/checkpointF{filters}B{batch}.model.keras'
        savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                monitor='val_loss',
                                                                save_best_only=True,
                                                                mode='min'
                                                                )
        model = model_net_MobileNetv2.compile(optimizer=optimAdam)
        EPOCHS = 100
        # Launch an experiment
        wandb.init(
            project="new_traning_dataset",
            name= f"{date.today()}|F{filters} B{batch}",
            config={
                "epoch": EPOCHS
            },
        )
        config = wandb.config
        # Add WandbMetricsLogger to log metrics
        wandb_callbacks =WandbMetricsLogger()

        model_history = model.fit(orig_train_batches,
                                  epochs=config.epoch,
                                  verbose = 0,
                                  validation_data=vald_batches,
                                  callbacks=[wandb_callbacks,savemodel_callback,DisplayCallback()]  #+EarlyStopping_callback
                                  )
        # Mark the run as finished
        wandb.finish()
