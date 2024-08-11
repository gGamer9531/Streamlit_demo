import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update this with your model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set up the Streamlit app layout
st.title("LLaMA 3.1 8B Question Answering")
st.write("Ask me anything!")

# Input box for user questions
user_input = st.text_input("Your Question:")

if user_input:
    # Encode the input and generate a response
    inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the response
    st.write("LLaMA's Answer:")
    st.write(response)
