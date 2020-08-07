import pandas as pd

data = pd.read_excel("D:\Github Repos\Flipkart-GRID-Noise-Cancellation2\ASR-API\ASRTranscriptions2.xlsx")

func = lambda x: (x.split('.')[0]).split('_')[-1]

data['Filename'] = data['Filename'].apply(func)

data = data.astype({"Filename": int})

data = data.sort_values('Filename')
data.to_excel("D:\Github Repos\Flipkart-GRID-Noise-Cancellation2\ASR-API\ASRTranscriptions2_sorted.xlsx", index = False)