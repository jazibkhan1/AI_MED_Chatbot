import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "LostPanda/llama2-finetune-medical"  # Replace with your model's name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Streamlit app
st.title("Chat with Medical Chatbot")

st.write("Type your message and press 'Send' to chat with the model:")

input_text = st.text_input("Your Message")

if st.button("Send"):
    # Add user input to chat history
    st.session_state.chat_history.append(f"User: {input_text}")

    # Generate model response
    inputs = tokenizer.encode(" ".join(st.session_state.chat_history), return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("User:")[-1].strip()

    # Add model response to chat history
    st.session_state.chat_history.append(f"Model: {response}")

# Display chat history
for message in st.session_state.chat_history:
    st.write(message)


'''
# Streamlit app
import streamlit as st
st.title("Medical Chatbot")
input_text = st.text_input("Input Text")
if st.button("Send"):
    if input_text:
        # Placeholder for where the model inference would happen
        st.write("Generated Text:")
        st.write("This is where the generated text will appear.")
    else:
        st.write("Please enter some text to generate.")'''
