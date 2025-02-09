import matplotlib.pyplot as plt

# these are all train metrics
avg_acc = {}
avg_loss = {}
avg_top5 = {}
# validaiton versions 
val_avg_acc = {}
val_avg_loss = {}
val_avg_top5 = {} 

with open('train_output.txt', 'r') as fin:
    lines = fin.readlines()


# epochs 1-6 
first_end_idx = lines.index('((\n')
first_idxs = []
for idx in range(first_end_idx):
    if 'Epoch 1' in lines[idx]:
        first_idxs.append(idx)
assert len(first_idxs) == 16, "ERROR IN EPOCH 1-6 EPOCH 1 FINDING"


for epoch in range(1, 7):
    accs = []
    losses = []
    top5s = []

    if epoch%3 == 0:
        val_accs = []
        val_losses = []
        val_top5s = [] 

    for fi in first_idxs:
        line = lines[fi + (2*epoch)-1]
        #print(line)
        a, l, t5 = line.index('accuracy'), line.index('- loss'), line.index('- top_k_categorical')
        if epoch%3 == 0:
            va = line.index("- val_accuracy")
            vl = line.index("- val_loss")
            vt5 = line.index("- val_top_k_categorical")
            accs.append(eval(line[a+10:l-1]))
            losses.append(eval(line[l+8:t5-1]))
            top5s.append(eval(line[t5+30:va-1]))
            val_accs.append(eval(line[va+16:vl-1]))
            val_losses.append(eval(line[vl+12:vt5-1]))
            val_top5s.append(eval(line[vt5+34:].strip()))
        else: 
            accs.append(eval(line[a+10:l-1]))
            losses.append(eval(line[l+8:t5-1]))
            top5s.append(eval(line[t5+30:].strip()))

    avg_acc[epoch] = sum(accs)/len(accs)
    avg_loss[epoch] = sum(losses)/len(losses)
    avg_top5[epoch] =  sum(top5s)/len(top5s) 

    if epoch%3 == 0: 
        val_avg_acc[epoch] = sum(val_accs)/len(val_accs)
        val_avg_loss[epoch] = sum(val_losses)/len(val_losses)
        val_avg_top5[epoch] = sum(val_top5s)/len(val_top5s) 


# after epoch 18 
starti = lines.index('after epoch 18: \n')+1 
# every 7 lines describes 3 epochs of 1 model
# then every 16 models done it has 2 lines of validation failed and 1 line saying it's starting 
# every 7*16 + 3 = 115 lines, it's like a 3-epoch cycle.

for starte in range(18,36+1, 3): #starte goes up to 36 
    # every 3 epoch cycle starts
    for adde in range(1, 4): # starte + adde goes up to 39, which is correct 
        accs = []
        losses = []
        top5s = []
        
        for student in range(16):
            if starte==27:
                # there's one missing student in those 
                if student==15:
                    break 
            line = lines[starti + 1 + adde + 7*student]
            #print(starte, adde, student, starti + 1 + adde + 7*student, lines[starti+7*student]) 
            a, l, t5 = line.index("TRAIN ACC"), line.index("TRAIN LOSS"), line.index("TRAIN TOP5")
            accs.append(eval(line[a+20:l-27]))
            losses.append(eval(line[l+21:t5-27]))
            top5s.append(eval(line[t5+21:-27]))

        avg_acc[starte+adde] = sum(accs)/len(accs)
        avg_loss[starte+adde] = sum(losses)/len(losses)
        avg_top5[starte+adde] =  sum(top5s)/len(top5s)

    if starte == 27:
        # there was a missing student
        starti -= 7 

    starti += 114 # to next cycle 



# plot train curves
epochs = sorted(list(avg_acc.keys()))
accs = [avg_acc[e] for e in epochs]

plt.plot(epochs, accs)
plt.title("Graph of Train Top-1 Accuracy against Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Top-1 Accuracy")
plt.savefig("./figures/train_acc.svg") 
plt.show()


losses = [avg_loss[e] for e in epochs]

plt.plot(epochs, losses)
plt.title("Graph of Train Loss against Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.savefig("./figures/train_loss.svg") 
plt.show()


top5s = [avg_top5[e] for e in epochs]

plt.plot(epochs, top5s)
plt.title("Graph of Train Top-5 Accuracy against Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Top-5 Accuracy")
plt.savefig("./figures/train_top5.svg") 
plt.show() 



# print out evaluation results in case they're used (spoiler: they aren't used)
print("val_avg_acc =", val_avg_acc)
print("val_avg_loss =", val_avg_loss)
print("val_avg_top5 =", val_avg_top5)

'''
Output:
val_avg_acc = {3: 0.024450000000000006, 6: 0.0537375}
val_avg_loss = {3: 1.1023625000000001, 6: 2.11195}
val_avg_top5 = {3: 0.5303187500000001, 6: 0.642375}
'''

