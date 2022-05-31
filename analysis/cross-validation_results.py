'''
This script is for extracting and summarizing the results from the cross-validation of the neural network models. 
'''
# %%
import os
import numpy as np
import pandas as pd
from statistics import mean, stdev
import matplotlib.pyplot as plt

# %%
orig = pd.DataFrame(pd.read_csv(os.path.join("..", "models", "cv_nn1_epoch5_batchsize32_s224_nokupka_orig.csv")))
style = pd.DataFrame(pd.read_csv(os.path.join("..", "models", "cv_nn1_epoch5_batchsize32_s224_nokupka.csv")))

df = style

#############################
#### AVERAGE PERFORMANCE ####
#############################
# %%

gogh_precision = [float(df['cm'][i][110:114]) for i in range(len(df['cm']))]
gogh_recall = [float(df['cm'][i][120:124]) for i in range(len(df['cm']))]
gogh_f1 = [float(df['cm'][i][130:134]) for i in range(len(df['cm']))]
gogh_support = [float(df['cm'][i][142:144]) for i in range(len(df['cm']))]

monet_precision = [float(df['cm'][i][182:186]) for i in range(len(df['cm']))]
monet_recall = [float(df['cm'][i][192:196]) for i in range(len(df['cm']))]
monet_f1 = [float(df['cm'][i][202:206]) for i in range(len(df['cm']))]
monet_support = [float(df['cm'][i][214:216]) for i in range(len(df['cm']))]

goya_precision = [float(df['cm'][i][254:258]) for i in range(len(df['cm']))]
goya_recall = [float(df['cm'][i][264:268]) for i in range(len(df['cm']))]
goya_f1 = [float(df['cm'][i][274:278]) for i in range(len(df['cm']))]
goya_support = [float(df['cm'][0][286:288]) for i in range(len(df['cm']))]

accuracy = [float(df['cm'][i][347:351]) for i in range(len(df['cm']))]
accuracy_support = [float(df['cm'][i][359:361]) for i in range(len(df['cm']))]
macro_precision = [float(df['cm'][i][399:403]) for i in range(len(df['cm']))]
macro_recall = [float(df['cm'][i][409:413]) for i in range(len(df['cm']))]
macro_f1 = [float(df['cm'][i][419:423]) for i in range(len(df['cm']))]
macro_support = [float(df['cm'][i][431:433]) for i in range(len(df['cm']))]
weighted_precision = [float(df['cm'][i][471:475]) for i in range(len(df['cm']))]
weighted_recall = [float(df['cm'][i][481:485]) for i in range(len(df['cm']))]
weighted_f1 = [float(df['cm'][i][491:495]) for i in range(len(df['cm']))]
weighted_support = [float(df['cm'][i][503:505]) for i in range(len(df['cm']))]


# %%
variable = goya_precision
print(mean(variable))
print(stdev(variable))







# %%
##################
#### PLOTTING ####
##################
# %%

epochs = 5

acc = [eval(df['history'][i])['accuracy'] for i in range(epochs)]
val_acc = [eval(df['history'][i])['val_accuracy'] for i in range(epochs)]

loss = [eval(df['history'][i])['loss'] for i in range(epochs)]
val_loss = [eval(df['history'][i])['val_loss'] for i in range(epochs)]

# %%
# Visualize performance
fig, axes = plt.subplots(2, figsize=(10, 8), sharex=True)
#plt.style.use("fivethirtyeight")
#plt.figure()
for i in range(len(loss)):
    axes[0].plot(np.arange(0, epochs), acc[i], color='tab:red', alpha=0.7, linewidth=2)
    axes[0].plot(np.arange(0, epochs), val_acc[i], color='tab:blue', alpha=0.7, linewidth=2)
    axes[0].set_title("Training Accuracy", fontsize=20)
    axes[0].set_ylabel("Accuracy", fontsize=15)
for i in range(len(loss)):
    axes[1].plot(np.arange(0, epochs), loss[i], color='tab:red', alpha=0.7, linewidth=2)
    axes[1].plot(np.arange(0, epochs), val_loss[i], color='tab:blue', alpha=0.7, linewidth=2)
    axes[1].set_title("Training Loss", fontsize=20)
    axes[1].set_ylabel("Loss", fontsize=15)
    axes[1].set_xlabel("Epoch #", fontsize=15)

fig.legend(["train", "val"], loc='center right', bbox_to_anchor=(1, 0.5), framealpha=0)


plt.tight_layout()
plt.show()

fig.savefig("../plots/cv_performance_style.png", transparent=True)

