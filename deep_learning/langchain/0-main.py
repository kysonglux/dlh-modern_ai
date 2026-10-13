#!/usr/bin/env python3

initialize_hf_llm = __import__('0-initialize_hf').initialize_hf_llm


llm = initialize_hf_llm("google/flan-t5-base", max_tokens=50)
prompt = "Answer the following question in one sentence: Who is Albert Einstein?"

print(llm)
response = llm.invoke(prompt)
print(response)