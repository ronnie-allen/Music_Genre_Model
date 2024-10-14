import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

# Load Model
def load_model(): 
    model=tf.keras.models.load_model('./Trained_model.h5')
    return model

# Load and Preprocess file
def load_and_preprocess(file_path,target_shape=[210,210]):
    data=[]
    audio_data,sample_rate= librosa.load(file_path,sr=None)
    chunk_duration=4
    overlap_duration=2

    chunk_sample=chunk_duration * sample_rate
    overlap_sample=overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_sample) / (chunk_sample - overlap_sample))) + 1

    # Iterate over chunks
    for i in range(num_chunks):
                        start = i * (chunk_sample - overlap_sample)
                        end = start + chunk_sample
                        chunk = audio_data[start:end]

                        # Check if the chunk is valid (not empty)
                        if len(chunk) < chunk_sample:
                            continue

                        # Compute the Mel spectrogram
                        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

                        # Resizing the Mel spectrogram
                        melspectrogram = resize(np.expand_dims(melspectrogram, axis=-1), target_shape)

                        # Append to data
                        data.append(melspectrogram)
    return np.array(data)


#  Model Prediction
def model_prediction(X_Test):
    model=load_model()
    y_p=model.predict(X_Test)
    pred_cat= np.argmax(y_p,axis=1)
    unique_elements,counts=np.unique(pred_cat,return_counts=True)
    max_count= np.max(counts)
    max_elements= unique_elements[counts==max_count]
    return max_elements[0]

# Streamlit UI with Aesthetic Improvements
import streamlit as st

# Sidebar Configuration
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ("Home", "Prediction", "About"))

# Custom CSS for Aesthetic Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a1a;  /* Dark grey background */
        color: #FFFFFF;  /* White text */
    }
    .sidebar .sidebar-content {
        background-color: #2E2E2E;  /* Slightly darker grey sidebar */
        color: #FFFFFF;  /* White text for visibility */
    }
    h1, h2, h3, h4, h5, h6, .markdown-text-container {
        color: #FFFFFF;  /* White headers */
    }
    .css-1d391kg p {
        color: #DDDDDD;  /* Light grey text for better readability */
    }
    .stButton>button {
        background-color: #004FC7;  /* Custom button color */
        color: #FFFFFF;
        border-radius: 10px;  /* Rounded buttons */
    }
    .stButton>button:hover {
        background-color: #002F6C;  /* Darker hover effect */
    }
    .stSpinner>div {
        border-color: #004FC7 transparent transparent transparent;  /* Spinner styling */
    }
    .sidebar .css-1d391kg { 
        color: #FFFFFF;  /* White text for the sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Page Logic
if app_mode == "Home":
    st.markdown('''## Welcome to the,\n ## Music Genre Classification System! 🎶🎧''')
    image_path = "./Home_Banner.png"
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
    **Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**
    
    ### How It Works
    1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
    2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
    3. **Results:** View the predicted genre along with related information.
    
    ### Why Choose Us?
    - **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
    - **User-Friendly:** Simple and intuitive interface for a smooth user experience.
    - **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.
    
    ### Get Started
    Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!
    
    ### About Us
    Learn more about the project, our team, and our mission on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.markdown("""
        ### About the Project
        Music. Experts have been trying for a long time to understand sound and what differentiates one song from another. How to visualize sound. What makes a tone different from another.
        
        This data hopefully can give the opportunity to do just that.

        ### About Dataset
        - **Genres:** A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (GTZAN dataset).
        - **Genres List:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.
        - **Images:** Visual representations for each audio file (Mel Spectrograms).
        - **2 CSV Files:** Containing extracted features from the audio files. Songs were split into 3-second audio chunks for more data.
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    
    if test_mp3 is not None:
        filepath = 'Music_Testing/' + test_mp3.name

        if st.button("Play Audio"):
            st.audio(test_mp3)

        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                X_test = load_and_preprocess(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
                st.markdown(f"## 🎵 Model Prediction: It's a 🎶 :red[{label[result_index]}] music! ##")
