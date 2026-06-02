# Face Recognition Attendance System

A desktop attendance system built with Python, Tkinter, OpenCV Haar Cascade, and LBPH face recognition.

## Features

- Register students with numeric IDs and names.
- Capture face images from the webcam.
- Train an LBPH face recognition model.
- Track faces from the webcam and mark daily attendance.
- Save attendance logs as CSV files.

## Requirements

- Python 3.11+
- Webcam access
- macOS, Windows, or Linux with a working OpenCV camera backend

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Run

```bash
python main.py
```

## Usage

1. Enter a numeric student ID and name.
2. Click `Take Images` and wait until face samples are captured.
3. Click `Train Model`.
4. Click `Track & Mark Attendance` to recognize a student and save attendance.

## Generated Files

The app creates these runtime files locally:

- `TrainingImage/` stores captured face images.
- `TrainingImageLabel/Trainner.yml` stores the trained OpenCV model.
- `StudentDetails/StudentDetails.csv` stores registered student data.
- `Attendance/Attendance_YYYY-MM-DD.csv` stores attendance logs.

These files are ignored by Git because they can contain personal data and generated model output.

## Notes

- Keep `haarcascade_frontalface_default.xml` in the project root.
- If tracking fails with an invalid model error, click `Train Model` again to rebuild the model.
