import streamlit as st
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#
image_path = r"Image/image.JPG"
image = Image.open(image_path)

st.set_page_config(page_title="English To Hindi Language Translator App", layout="centered")
st.image(image, caption='English To Hindi Language Translator')
# page header
st.title(f"English Text to Hindi Translation App")
with st.form("Prediction_form"):
   text = st.text_input("Enter text here")
   #st.title(text)
   #
   submit = st.form_submit_button("Translate Text to Hindi")
   #
   if submit:
        tokenizer = AutoTokenizer.from_pretrained(r"model_files/tk",use_auth_token=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(r"model_files/md",use_auth_token=False)
        inputs = tokenizer(text, return_tensors="pt")

        translated_tokens = model.generate(**inputs, 
                                           forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"], 
                                           max_length=100)
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(result)
        # output header
        st.header("Translated Text")
        # output results
        st.success(f"Translated Text : {result}")
