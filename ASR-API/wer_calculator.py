from wer import wer
import pandas as pd

ASR_data = pd.read_excel("C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/RandomASRTranscriptions.xlsx")
true_data = pd.read_csv("C:/Users/nihal/Downloads/FlipkartNoiseCancel/Round3/Key Guidelines and Audio Recording Updated/Flipkart-GRID-Noise-Cancellation/ASR-API/Truetranscriptions.csv")
print(true_data.head())
#print(ASR_data.head())

# From each row of ASR_data, extract filename and transcript. Then compare and find a row in true_data that has the same filename and extract transcript.
# Now compute WER between the two transcripts. Add to total_wer. After the loop, divide by number of files (average WER).

total_wer = 0
for ind,ser in ASR_data.iterrows():
    #print("ind: ", ind)
    print()
    print("ser: ", ser.iloc[0])
    try:
        ASR_filename = int(ser.iloc[0])
    except:
        break

    ASR_transcript = str(ser.iloc[1])

    true_transcript = str(true_data[true_data["Audio ID"] == int(ASR_filename)]['Transcription'].values[0])
    #print(type(true_transcript))
    wer_here = wer(true_transcript, ASR_transcript)
    print("wer: ", wer_here)
    total_wer += wer_here

avg_wer = total_wer/len(ASR_data)
print(avg_wer)