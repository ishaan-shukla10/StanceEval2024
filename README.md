1) First run the data.py script to create DataLoaders to be used in model.py
2) Run the model.py script on different models and save the weights of the model under a .pt file extension
3) Get the best epochs of top 3 models, i.e. epochs in which model has the highest validation F1 score
4) Resultant from above steps would be 3 .pt files of 3 best epochs of models, which in our case were 'aubmindlab/bert-base-arabertv02-twitter', 'CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth' and 'UBC-NLP/MARBERT'
5) Note that the 'UBC-NLP/MARBERT' model was trained as a Multi-Task Learning model, whereas the other two were trained as Single-Task Learning models
6) The model 'aubmindlab/bert-base-arabertv02-twitter' was trained on unprocessed data, i.e. data which contained words like 'URL' and other emojis, whereas the other 2 models were trained on processed data
7) Then using the 3 .pt files, draw inferences using the inference.py script to get 3 separate .csv files, which would be the predictions by each of the models separately
8) Then using these 3 .csv files, perform an ensemble using ensemble.py script to get the final output, a .csv file after a hard ensemble 
