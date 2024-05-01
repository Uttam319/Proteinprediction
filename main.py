import numpy as np
from time import time
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset_reshaped, split_dataset, get_resphaped_dataset_paper, get_cb513, is_filtered
import model
import pickle

filtered = is_filtered()

do_log = True
stop_early = False
show_plots = True

start_time = timer()

print("Collecting Dataset...")

if filtered:
    # Split the dataset in 0.8 train, 0.1 test, 0.1 validation with shuffle (optionally seed)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_dataset_reshaped(seed=100)
else:
    # Slit the dataset with the same indexes used in the paper (Only for CullPDB6133 not filtered)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = get_resphaped_dataset_paper()

end_time = timer()
print("\n\nTime elapsed getting Dataset: " + "{0:.2f}".format((end_time - start_time)) + " s")

if filtered:
    print("Using CullPDB Filtered dataset")

net = model.CNN_model()

start_time = timer()

history = None

call_b = [model.checkpoint]

if filtered:
    logDir = "CullPDB_Filtered/{}".format(time())
else:
    logDir = "CullPDB/{}".format(time())

if do_log:
    call_b.append(callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, write_graph=True))

if stop_early:
    call_b.append(model.early_stop)

history = net.fit(X_train, Y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=call_b)

end_time = timer()
print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")

scores = net.evaluate(X_test, Y_test)
print("Loss: " + str(scores[0]) + ", Accuracy: " + str(scores[1]) + ", MAE: " + str(scores[2]))
#print(scores)

CB_x, CB_y = get_cb513()

cb_scores = net.evaluate(CB_x, CB_y)
print("CB513 -- Loss: " + str(cb_scores[0]) + ", Accuracy: " + str(cb_scores[1]) + ", MAE: " + str(cb_scores[2]))

# Pickle the history object without callbacks
history_without_callbacks = {key: value for key, value in history.history.items() if not key.startswith('val_')}

pickle_out = open("lasthistory.pickle","wb")
pickle.dump(history_without_callbacks, pickle_out)
pickle_out.close()

if show_plots:
    from plot_history import plot_history
    plot_history(history)
