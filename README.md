
# Emotion Based Music Recommendation

## Motive

The motive behind an Expression-based music recommendation project is to provide personalized music recommendations that resonate deeply with users on an emotional level. Instead of relying solely on algorithms based on genres, beats per minute, or popularity, this project aims to understand the nuanced emotions and moods of users and recommend music that aligns with those feelings.

## Dataset

The FER 2013 dataset, often referred to as FER (Facial Expression Recognition), is a widely used benchmark dataset in the field of computer vision and facial expression recognition. It contains grayscale images of faces categorized into seven different emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. Since this dataset is given to us in grayscale, I had to batch convert the images to RGB as VGG19 only supports RGB inputs.

## Haar Cascades

Haar cascades are a machine learning object detection method used to identify objects or features within images or video frames. They work by training on positive and negative images. Positive images contain the target object (e.g., pedestrians), while negative images do not.

The training process involves creating a cascade of classifiers, where each classifier is a binary decision-making unit. These classifiers are organized in a cascade, with the easy-to-decide features evaluated first. Features are rectangular patterns, often black and white, which are shifted and scaled over the image to extract relevant information.

During detection, the Haar cascades slide over the image and apply the learned features to identify regions of interest that may contain the target object. If a region passes through all cascade stages, it is marked as a positive detection. The process is fast and efficient, making it suitable for real-time object detection in videos and images.

In this project, Haar cascades were used in conjunction with OpenCV to detect human faces in video frames, producing regions of interest for further analysis.

## Process

1. **Load Emotion Detection Model**:
   - A pre-trained convolutional neural network (CNN) model for emotion detection is loaded. CNNs are widely used for image classification tasks due to their effectiveness in capturing spatial dependencies in images.

2. **Capture Video**:
   - The script initializes the webcam feed to capture live video frames. This live feed will be processed for emotion detection.

3. **Data Preprocessing**:
   - Before passing the face images to the emotion detection model, they undergo preprocessing to ensure compatibility with the model's input requirements.
   - Preprocessing typically involves resizing the images to a fixed size (e.g., 48x48 pixels), converting them to RGB, and normalizing pixel values to a specific range (e.g., [0, 1]).
   - These preprocessing steps standardize the input data and help improve the model's performance by reducing variations and noise in the input images.

4. **Face Detection and Emotion Recognition**:
   - Haar cascade classifier is used for face detection. Although not as sophisticated as deep learning-based methods, Haar cascades are fast and suitable for real-time applications.
   - Once faces are detected, they are preprocessed and passed through the emotion detection model. Preprocessing typically involves resizing the image to a fixed size and converting it to the appropriate format for model input.
   - The emotion detection model predicts the dominant emotion in each detected face. This prediction forms the basis for selecting the appropriate song to play.

5. **Audio Playback**:
   - Based on the predicted emotion, the corresponding song is played using Pygame. Music has a profound impact on emotions, making it an ideal medium for enhancing the user experience.
   - Separate threads are used for playing audio to prevent blocking the main thread, ensuring smooth operation of the emotion detection process.
   - If a different emotion is detected subsequently, the currently playing song is stopped, and the new song corresponding to the new emotion is played. This dynamic adjustment ensures that the music aligns with the user's current emotional state.

6. **Display Output**:
   - Rectangles are drawn around detected faces, and text indicating the predicted emotion is overlaid on each face. This visual feedback provides real-time information about the emotion recognition process.
   - The processed video feed with emotion annotations is displayed in a window, allowing users to see the system's response to their facial expressions.

![Output Window](assets\happy.jpg)

## Model used

I have used the VGG19 Model. (The model weights and json file can be found at :
<https://drive.google.com/drive/folders/1Vyd3Yd8N1kJimkwb-zTKM1qgiFKMrLdT?usp=drive_link>)

1. **Architecture**: VGG-19 is characterized by its deep architecture, consisting of 19 layers (hence the name). It follows a straightforward and uniform architecture pattern with a stack of convolutional layers followed by max-pooling layers. The convolutional layers use small receptive fields (3x3) with a stride of 1 and zero-padding to maintain the spatial resolution.

2. **Layer Configuration**: The network architecture of VGG-19 typically consists of alternating convolutional layers with ReLU (Rectified Linear Unit) activations and max-pooling layers. The deeper layers capture increasingly complex patterns and features in the input images. I have frozen the last 10 layers and have added a dense layer with 7 neurons to classify the outputs.

3. **Parameter Efficiency**: Despite its depth, VGG-19 has relatively few parameters compared to other contemporary architectures like ResNet or Inception. This parameter efficiency is due to the exclusive use of small 3x3 filters and the absence of complicated modules like residual connections or inception modules.

4. **Transfer Learning**: VGG-19 is commonly used as a base model for transfer learning tasks. Transfer learning involves leveraging pre-trained models trained on large datasets (such as ImageNet) and fine-tuning them on specific tasks with smaller datasets. The features learned by VGG-19 on ImageNet can be transferred to new tasks, allowing for efficient training even with limited data.

## Limitations and Future Scope

1. The detection and recognition of multiple faces and expressions is still a task, typically since the songs are hard coded

2. As mentioned above, the songs are hard coded. This is self explanatory and a more robust method where perhaps the user can select a song themselves with respect to their moods would be preferable. These selections can be run through various clustering or similarity search algorithms to also recommend similar songs to the user with respect to the moods

3. There is a continuous scope for enhancing the performance of facial expression recognition models. This includes exploring more advanced neural network architectures, incorporating attention mechanisms, and leveraging techniques like data augmentation and domain adaptation to improve model generalization and robustness

4. Integrating multiple modalities such as audio, text, and physiological signals with facial expressions can provide richer contextual information for emotion recognition. Future research could focus on developing multimodal models that fuse information from different modalities to achieve more accurate and nuanced emotion recognition.

Feel free to experiment with my project. The folder link has been provided, where the model weights and json are available to download.
