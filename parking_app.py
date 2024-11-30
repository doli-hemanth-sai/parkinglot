import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import cv2

# Initialize YOLO model
model = YOLO("parking_final.pt")

# Define Streamlit app
def main():
    st.title('Parking Lot Recognition')

    # Sidebar for file upload and type selection
    st.sidebar.title('Input Options')
    choice = st.sidebar.radio('Choose Input Type', ('Image', 'Video'))

    # Main content based on choice
    if choice == 'Image':
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Perform prediction
            results = model.predict(image)
            result=results[0]
            # Display results
            st.subheader("Output:")
            st.image(Image.fromarray(result.plot()[:,:,::-1]),caption='Processed Image', use_column_width=True)

    elif choice == 'Video':
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
            st.warning("Video may take time for processing....!")
            # Open the video
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create a video writer to save the processed frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

            # Process the video frames and save them
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform prediction on the frame
                results = model.predict(frame)
                result = results[0]

                # Write the processed frame to the output video
                out.write(cv2.cvtColor(result.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR))

            # Release the video capture and writer objects
            cap.release()
            out.release()

            # Display the output video
            st.video('output_video.mp4')

if __name__ == '__main__':
    main()
