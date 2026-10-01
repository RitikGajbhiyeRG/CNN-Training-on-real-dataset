# AC & Fan Image Classifier

A Streamlit app that classifies an uploaded image as one of the classes in the
trained CNN checkpoint (`Ac` and `Fan`), based on the model saved from Google Colab.

## Files
- `app.py`: Streamlit interface and prediction code.
- `requirements.txt`: Python dependencies.

## Before deploying
1. Keep your downloaded `ac_fan_cnn.pth` model in Google Drive.
2. In Google Drive, right-click the model -> Share -> General access ->
   **Anyone with the link** -> Viewer.
3. Copy the file ID from the share URL. For example, in
   `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`, the ID is `FILE_ID`.
4. Open `app.py` and replace `PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE` in
   `DRIVE_FILE_ID` with your actual file ID.
5. Commit `app.py`, `requirements.txt`, and this README to your GitHub repository.

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io/
2. Connect your GitHub account and select your repository.
3. Set the main file path to `app.py`.
4. Deploy.

The first prediction downloads the model from Google Drive, so the file must remain
accessible to anyone with the link. Do not upload private or sensitive files publicly.

## Important
The app architecture and image normalization are matched to the provided notebook.
The model was trained for one epoch and reported 74.51% test accuracy in the shown
Colab run. That accuracy is only a result of that particular split and does not
guarantee performance on new images.
