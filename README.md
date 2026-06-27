# Face Recognition Attendance System

A desktop attendance system built with Python, Tkinter, OpenCV Haar Cascade, and LBPH face recognition.

## Features

- Register students with numeric IDs and names.
- Capture face images from the webcam.
- Train an LBPH face recognition model.
- Track faces from the webcam and mark daily attendance.
- Save attendance logs as CSV files.

## Download (no setup required)

Don't want to install Python? Download the ready-to-run app:

**➡️ https://github.com/Hinsane5/ComputerVisionAttendance/releases/latest**

1. Download the file for your computer:
   - **Windows** → `AttendanceSystem-Windows.zip`
   - **Mac** → `AttendanceSystem-macOS.zip`
2. Unzip it.
3. Double-click `AttendanceSystem` to launch.

### First launch: bypass the security warning

The app is not code-signed, so your system shows a one-time warning the first
time you open it. This is expected — here is how to get past it:

- **Mac:** right-click (or Control-click) the app → **Open** → click **Open** again.
- **Windows:** on the "Windows protected your PC" screen, click **More info** → **Run anyway**.

> **Mac says "AttendanceSystem is damaged and can't be opened"?** That is macOS
> blocking the download, not a real problem. Open the Terminal app and run this
> once (drag the app onto the Terminal window to fill in the path):
>
> ```bash
> xattr -dr com.apple.quarantine /path/to/AttendanceSystem.app
> ```
>
> Then double-click the app again.

> The app asks for camera access on first use — click **Allow**. Your data
> (students, model, attendance logs) is saved in an `AttendanceSystem` folder in
> your home directory.

## Run from source (for developers)

The sections below are only needed if you want to run or modify the code
directly instead of using the downloadable app above.

### Requirements

- Python 3.11+
- Webcam access
- macOS, Windows, or Linux with a working OpenCV camera backend

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### Run

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
