# SAFE_SCAN — Document Authenticity Verification System

**SAFE_SCAN** is an AI-powered document verification system designed to detect forgery and validate the authenticity of official documents. The system currently supports passport verification for Azerbaijan and ID card verification for Estonia.

## Project Objective

The main goal of this system is to identify whether an uploaded document image is valid or forged. The system checks for the presence of key elements (face, signature, document structure), extracts textual information, analyzes metadata for tampering signs, and verifies document numbers against a local database.

## Supported Document Types

- **Azerbaijan Passport**
- **Estonia ID Card**

Each type includes custom logic tailored to the visual and textual structure of the respective country's documents.

## Features

- Object detection using **YOLOv5** to identify critical document elements
- Text extraction via **Tesseract OCR**
- Expiry date analysis and forgery detection
- Metadata inspection with **ExifTool** for signs of image manipulation
- Local database validation for document numbers
- Simple web interface built with **Flask** and **Bootstrap**

## Sample Interface

The frontend provides a clean, responsive form allowing users to:
- Select the document's country
- Upload an image file for verification
- View a step-by-step summary of the validation results

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SAFE_SCAN.git
   cd SAFE_SCAN

2. **Create a virtual environment (optional)**
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Tesseract-OCR**
Download and install from: https://github.com/tesseract-ocr/tesseract
Update the path to tesseract.exe in app.py

4. **Download ExifTool**
Download from: https://exiftool.org/
Set the path to exiftool.exe in the processor classes

5. **Ensure your YOLOv5 model weights are in place**
Update paths in app.py and processor classes accordingly

6. **Run the application**
python app.py
Open http://127.0.0.1:5000 in your browser.
