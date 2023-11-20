import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tempfile
import os


class SCNN(nn.Module):
    def __init__(self , input_size , massage_kernel = 9 , pretrained=True):
        super(SCNN , self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size , massage_kernel)
        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background, 1, 1, 1, 1]))
        self.bce_loss = nn.BCELoss()

    def net_init(self,input_size , ms_ks):
        input_w , input_h = input_size
        self.fc_input_size = 5 * int(input_w/16) * int(input_h/16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        #replace the standard convs with dilated convs
        for i in [34 , 37 , 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(conv.in_channels , conv.out_channels  , conv.kernel_size ,
                                    stride=conv.stride , padding = tuple(p*2 for p in conv.padding) ,
                                    dilation=2 , bias = (conv.bias is not None))
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        #scnn unit
        self.layer1 = nn.Sequential(
            nn.Conv2d(512 , 1024 ,3 ,  padding=4 , dilation=4 , bias=False) ,
            nn.BatchNorm2d(1024) ,
            nn.ReLU() ,
            nn.Conv2d(1024 , 128 , 1 , bias=False),
            nn.ReLU()
        )

        # add message passing
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module("up_down" , nn.Conv2d( 128 , 128 , (1 , ms_ks) , padding=(0,ms_ks//2) , bias=False     ))
        self.message_passing.add_module("down_up" , nn.Conv2d(128,128,(1,ms_ks) , padding=(0,ms_ks//2) , bias=False))
        self.message_passing.add_module('left_right',nn.Conv2d(128,128,(ms_ks , 1) , padding=(ms_ks//2 , 0) , bias=False))
        self.message_passing.add_module("right_left" , nn.Conv2d(128,128,(ms_ks , 1) , padding=(ms_ks//2 , 0) , bias=False))

        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1) ,
            nn.Conv2d(128,5,1)
        )
        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1) ,
            #dimension reducion by 2
            nn.AvgPool2d(2,2) ,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size , 128) ,
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )

    def message_passing_forward(self , x):
        Vertical=[True , True , False , False ]
        Reverse = [False , True , False , True]
        for ms_conv  , v , r in zip(self.message_passing , Vertical , Reverse):
            x = self.message_passing_once(x  , ms_conv , v , r)
        return x

    def message_passing_once(self,x  , ms_conv , vertical=True , reverse=True):
        nB , C , H , W = x.shape
        if vertical :
            slices =[  x[: , : , i : (i+1) , : ] for i in range(H)  ]
            dim=2
        else :
            slices = [ x[: , : , : , i: (i+1)] for i in range(W) ]
            dim=3
        if reverse :
            slices = slices[::-1]

        #then each slice convole with the conv layer and add to the previous layer
        out = [ slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i]+F.relu(ms_conv(out[i-1])))

        if reverse :
            out = out[::-1]
        #concatenate the tensors with the dimension
        return torch.cat(out , dim=dim)

    def forward(self,img , seg_img=None , exist_gt=None):
        #inference thorught the vgg16 backbone net
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        #then to obtain the original image size need to upsample by 8
        seg_pred = F.interpolate(x  , scale_factor=8 , mode='bilinear' , align_corners=True)
        x = self.layer3(x)
        x = x.view(-1  , self.fc_input_size)
        exist_pred = self.fc(x)

        if seg_img is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred , seg_img.long().squeeze(1))
            loss_exist = self.bce_loss(exist_pred.float() , exist_gt.float())
            #nned to pay more attention on the segmanetation loss and weight should be high
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist

        else:
            loss_seg = torch.tensor(0,dtype=img.dtype , device=img.device)
            loss_exist = torch.tensor(0,dtype=img.dtype , device=img.device)
            loss = torch.tensor(0,dtype=img.dtype , device=img.device)

        return seg_pred , exist_pred , loss_seg , loss_exist , loss



def load_model(model_path):
    # Create an instance of SCNN model
    model = SCNN((800, 288), pretrained=False)
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def process_image(image_path):
    # Read the image using OpenCV
    original_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform the image and add batch dimension
    image_tensor = transform(img_rgb).unsqueeze(0)
    return original_img, image_tensor

def convert_to_lane_markings(seg_pred, labels):
    seg_pred = seg_pred.squeeze(0)  # Remove batch dimension
    seg_pred = torch.argmax(seg_pred, dim=0)  # Get the class index for each pixel
    seg_pred = seg_pred.cpu().numpy()  # Convert to numpy array

    lane_markings = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(labels):
        lane_markings[seg_pred == class_idx] = color

    return lane_markings

def overlay_lane_markings(original_image, lane_markings):
    # Resize lane_markings to match the size of original_image
    lane_markings_resized = cv2.resize(lane_markings, (original_image.shape[1], original_image.shape[0]))

    # Ensure original_image is in RGB format
    if len(original_image.shape) == 2 or original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Overlay the lane markings on the original image
    overlay = cv2.addWeighted(original_image, 1, lane_markings_resized, 0.5, 0)
    return overlay


def predict(model, image_tensor, original_image, labels):
    with torch.no_grad():
        seg_pred, _, _, _, _ = model(image_tensor)
    lane_markings = convert_to_lane_markings(seg_pred, labels)
    overlay_image = overlay_lane_markings(original_image, lane_markings)
    return overlay_image

def extract_frames(video_path, temp_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(temp_folder, f"frame{count:05d}.jpg"), image)     
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return count  # Return the number of extracted frames

def create_video_from_frames(frame_folder, output_video_path, fps=30):
    frame_paths = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    if not frame_paths:
        return
    
    # Read the first frame to determine the size
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_path in frame_paths:
        video.write(cv2.imread(frame_path))

    video.release()


# Load your model outside of the main loop to avoid reloading on every interaction
model = load_model('model_state_dict.pth')

# Streamlit webpage layout
st.title("Lane Detection App")
st.write("Upload an image to detect lanes.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert('RGB')
    image.save("temp.jpg")  # Save the uploaded image to process
    original_img, image_tensor = process_image("temp.jpg")
    labels = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    # Predict
    result_image = predict(model, image_tensor, original_img, labels)

    # Display the image
    st.image(result_image, caption='Processed Image', use_column_width=True)



uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi"], key="video_uploader")

if uploaded_video is not None:
    with tempfile.TemporaryDirectory() as temp_folder:
        # Save the uploaded video
        video_path = os.path.join(temp_folder, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Extract frames
        extract_frames(video_path, temp_folder)

        # Process each frame
        for frame_file in sorted(os.listdir(temp_folder)):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(temp_folder, frame_file)
                original_img, image_tensor = process_image(frame_path)
                labels = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)] 
                result_image = predict(model, image_tensor, original_img, labels)
                cv2.imwrite(frame_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

        # Recombine frames into a video
        output_video_path = "processed_video.mp4"
        create_video_from_frames(temp_folder, output_video_path)

        # Display or offer download of the video
        st.video(output_video_path)
