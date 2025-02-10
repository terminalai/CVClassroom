import matplotlib.pyplot as plt

#valid_accs = {0: 0.0823, 6: 0.0737, 12: 0.0866, 18: 0.0804, 21: 0.0767, 24: 0.0841, 27: 0.0773, 30: 0.0853, 33: 0.0804, 36: 0.0816, 39: 0.0755}
#valid_top5_accs = {0: 0.6489, 6: 0.426, 12: 0.407, 18: 0.4045, 21: 0.4009, 24: 0.4002, 27: 0.399, 30: 0.4052, 33: 0.4082, 36: 0.4045, 39: 0.4076}

valid_accs = {0: 0.08225905150175095, 6: 0.0736648216843605, 12: 0.08655616641044617, 18: 0.08041743189096451, 21: 0.07673419266939163, 24: 0.08410067856311798, 27: 0.07734806835651398, 30: 0.08532842248678207, 33: 0.08041743189096451, 36: 0.0816451832652092, 39: 0.07550644874572754}
valid_top5_accs = {0: 0.6488643288612366, 6: 0.4260282516479492, 12: 0.4069981575012207, 18: 0.4045426547527313, 21: 0.40085941553115845, 24: 0.4002455472946167, 27: 0.3990178108215332, 30: 0.40515652298927307, 33: 0.4082258939743042, 36: 0.4045426547527313, 39: 0.40761202573776245}

test_accs = {0: 0.08742693811655045, 6: 0.08369605988264084, 12: 0.08581022173166275, 18: 0.08444223552942276, 21: 0.08680512011051178, 24: 0.08419350534677505, 27: 0.08294988423585892, 30: 0.08680512011051178, 33: 0.08581022173166275, 36: 0.08792439103126526, 39: 0.08742693811655045}
test_top5_accs = {0: 0.6577540040016174, 6: 0.44161173701286316, 12: 0.4177341163158417, 18: 0.41089415550231934, 21: 0.4145006835460663, 24: 0.42071881890296936, 27: 0.41251087188720703, 30: 0.40865564346313477, 33: 0.40952616930007935, 36: 0.4072876572608948, 39: 0.4061684012413025}


# plot valid results 
epochs = sorted(list(valid_accs.keys())) 
fig, ax1 = plt.subplots() 
color = 'tab:green' # accs first 
ax1.set_xlabel('epoch') 
ax1.set_ylabel('valid top-1 accuracy', color=color) 
ax1.plot(epochs, [valid_accs[e] for e in epochs], color=color) 
ax1.tick_params(axis='y', labelcolor=color) 

ax2 = ax1.twinx() # second y axis for top5 accs 

color = 'tab:blue' 
ax2.set_ylabel('valid top-5 accuracy', color=color) 
ax2.plot(epochs, [valid_top5_accs[e] for e in epochs], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title("Graph of Validation Accuracies against Epoch")

fig.tight_layout() # to ensure the right y label isn't clipped 
plt.savefig("./students/figures/valid_accs.svg")
plt.show() 





# plot test results 
epochs = sorted(list(test_accs.keys()))
fig, ax1 = plt.subplots()
color = 'tab:green' # accs first
ax1.set_xlabel('epoch')
ax1.set_ylabel('test top-1 accuracy', color=color)
ax1.plot(epochs, [test_accs[e] for e in epochs], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() # second y axis for top5 accs

color = 'tab:blue'
ax2.set_ylabel('test top-5 accuracy', color=color)
ax2.plot(epochs, [test_top5_accs[e] for e in epochs], color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title("Graph of Test Accuracies against Epoch")

fig.tight_layout() # to ensure the right y label isn't clipped  
plt.savefig("./students/figures/test_accs.svg")
plt.show() 


