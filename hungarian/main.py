from dataloader import orig_train_batches, vald_batches, single_batch
from callback import DisplayCallback# ,EarlyStopping_callback, savemodel_callback
import wandb
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
import sys

if __name__=="__main__":
    choice = int(sys.argv[1])
    if choice == 0:
        from hungarian import model
    if choice == 1:
        from mobilevit import model
    EPOCHS = 100
    # Launch an experiment
    option = {0:'V3', 1:'VIT'}
    wandb.init(
        project="hungarian",
        name= f"'hungarian{option[choice]}'|{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    # # Add WandbMetricsLogger to log metrics
    wandb_callbacks =WandbMetricsLogger()
    # model.load_weights('SavedModels/test.h5')
    model_history = model.fit(orig_train_batches,
                              epochs=config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[wandb_callbacks,DisplayCallback()]  #EarlyStopping_callback, savemodel_callback
                              )
    model.save(f'SavedModels/mobile{option[choice]}.h5')
    # Mark the run as finished
    wandb.finish()
