import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu
# x = np.arange(0,4*np.pi,0.1)   # start,stop,step
# y = np.sin(x)
#
# fig = plt.figure(figsize=(10, 10))
#
#
#
# for result in range(5,20):
#     for i in range(0,result):
#         print("Result and i",result,i)
#         ax = fig.add_subplot((result+1)//2, (result+1)//2, i+1)
#         ax.set_title("Check")
#         a = np.arange(i+1, i+17).reshape(4, 4)
#         ax.imshow(a, cmap='gray', alpha=0.6)
#     plt.savefig("test"+str(result)+"png")
#


real_caption =  "<start> a man riding his bike across the bridge that is over the river <end>"
result = ['<start>', 'a', 'man', 'in', 'shirts', 'scene', '<end>']


#result = ["<start>","a","man","<end>"]
#result  =" ".join(result)
print()
smoothie = SmoothingFunction().method4
score=bleu([real_caption],  result,smoothing_function=smoothie, weights=(0.25, 0.25, 0.25, 0.25))
print(score)
