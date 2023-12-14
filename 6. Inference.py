#
qna_df = pd.read_csv('QnA.csv')# QnA dataset 
rag_outputs = []
for index, row in qna_df.iterrows():
    question = row['question']
    predicted_answer = rag_pipeline(question)
    rag_outputs.append({
        'question': question,
        'answer': row['answer'],
        'predicted_answer': predicted_answer['result']
    })
result_df = pd.DataFrame(rag_outputs)
result_df.to_csv("rag_outputs.csv", index=False)