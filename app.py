import gradio as gr
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define prediction function
def predict_placement(IQ, CGPA, marks10, marks12, comm_skills):
    try:
        features = np.array([[int(IQ), float(CGPA), float(marks10), float(marks12), float(comm_skills)]])
        prediction = model.predict(features)
        return "🎓 Placed ✅" if prediction[0] == 1 else "❌ Not Placed"
    except Exception as e:
        return f"Error: {e}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_placement,
    inputs=[
        gr.Number(label="IQ"),
        gr.Number(label="CGPA"),
        gr.Number(label="10th Marks"),
        gr.Number(label="12th Marks"),
        gr.Number(label="Communication Skills (1–10)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Student Placement Predictor App",
    description="Enter academic and skill details to check if the student is likely to get placed.",
    theme="default"
)

# Launch app
if __name__ == "__main__":
    iface.launch()
