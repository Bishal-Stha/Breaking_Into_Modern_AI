from transformers import pipeline

generator = pipeline('text-generation', model='./nepal_cricket_model', tokenizer='./nepal_cricket_model')

# Start with a prompt related to the essay
prompt = input("Enter prompt: ")
print(generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text'])