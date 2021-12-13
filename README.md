# vnese-id-extractor

## Implemented by:

Ho Chi Minh City University of Technology - Control Engineering and Automation
1. Nguyen Ngoc Nhan
2. Thai Quang Nguyen
3. Pham Ngoc Tran
4. Nguyen Gia Khiem
5. Pham Binh Nguyen

## Youtube video tutorial

Link: https://youtu.be/4DhUrMDltvE

## How to run the code

### Step 1: Install anaconda, python and git
Check out this link: https://youtu.be/ZRC2nzP1w_4

### Step 2: Clone this repo
1. Open anaconda terminal, navigate to your directory
2. Run this command to clone the code:
```bash
git clone git@github.com:khiemnguyen240900/vnese-id-extractor.git
```
3. Then navigate to the code
```
cd vnese-id-extractor
```

### Step 3: Create conda environment and install requirements
1. Create conda environment with cpu or gpu
```bash
# Change directory to installation
cd installation

# for CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# for GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```
2. Install requirements
```bash
# for CPU
pip install -r requirements.txt

# for GPU
pip install -r requirements-gpu.txt
```
3. Back to the base folder
```bash
cd ..
```

### Step 4: Download and convert custom Yolo weights for id cards
1. Download my pre-trained weights at: https://drive.google.com/drive/folders/1TwrMzlOS2HuOv628ZOQeqANTQRpUapwh?usp=sharing
2. Put the weights file into: ./yolov4_card_detection/data/
3. Put the names file into: ./yolov4_card_detection/data/classes/
4. Open ./yolov4_card_detection/core/config.py, change line 15 to
```
__C.YOLO.BASE = os.getcwd().replace(os.sep, '/')
```
5. Convert the Yolo weights from darknet to tensorflow
```bash
cd yolov4_card_detection
python save_model.py --weights ./data/yolov4-cards.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4
cd .. 
```
6. Ensure the conversion is successful by checking ./yolov4_card_detection/checkpoints folder
7. Undo step 4.4 (change line 15 back to)
```
__C.YOLO.BASE = os.getcwd().replace(os.sep, '/') + "/yolov4_card_detection"
```
8. Note: to train with your custom data, check out this tutorial from theAIguys: https://youtu.be/mmj3nxGT2YQ

### Step 5: Run the code
1. Please read Appendix A for information about flags
2. Run the following code  (make sure you are in the base folder)
```bash
# using webcam with interactive mode
python main.py --weights /checkpoints/custom-416 --video 0 --interactive

# using phone camera through IP Webcam app and save the aligned image
python main.py --weights /checkpoints/custom-416 --camera_ip "YOUR-CAMERA-IP" --output
```
3. Extracted information and aligned image will be store in ./output folder

## Appendix A: Most useful flags
### alignment_process
  --alignment_process: show alignment process
    (default: 'false')
### camera ip
  --camera_ip: camera ip for external camera
### interactive
  --interactive: interactive mode in card alignment
    (default: 'false')
### output
  --output: to save aligned image to output folder
    (default: 'false')
### video
  --video: path to input video or set to 0 for webcam
    (default: '0')
### weights
  --weights: path to weights file
    (default: '/checkpoints/yolov4-416')
### Other flags
Other flags such as --iou, --model, --size,... can be read by running this command
```bash
python main.py --helpshort 
```
