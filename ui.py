import cv2
import numpy as np
from skimage import color, filters
import tkinter as tk
from customtkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from braintumour import *
import joblib

# Variables to hold the image and its path
saved_image = None
image_path = None
image_list = []
titles_list = []
current_index = -1

def open_image():
    global saved_image, image_path, image_list, titles_list, current_index
    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
    if file_path:
        # Open and display the image
        img = Image.open(file_path)
        img.thumbnail((128, 128))  # Resize the image to fit the Tkinter window
        img_tk = ImageTk.PhotoImage(img)

        # Save the image and its path for further use
        saved_image = img  # Save the image object
        image_path = file_path  # Save the image path
        
        # Update the label with the file path
        label_path.configure(text=file_path)  
        
        # Process the image and update the image list
        process_image(image_path)
        current_index = 0
        update_image_and_title()
        update_navigation_buttons()

# Tumor segmentation function
def segment_tumor(image):
    gray_image = image
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)
    return segmented_image



def process_image(image_path):
    global image_list, titles_list
    if image_path:
        # Load and process the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (400, 400))
        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel, Prewitt, and Roberts edge detection operators
        edges_sobel = filters.sobel(image)
        edges_prewitt = filters.prewitt(image)
        edges_roberts = filters.roberts(image)
        ksize = (5, 5)  # Kernel size (must be odd)
        sigma = 2.0  # Standard deviation of the Gaussian kernel
        blurred_image = cv2.GaussianBlur(image, ksize, sigma)
        segmented_image = segment_tumor(image)
        _, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        cmap_image = plt.cm.gray(thresh_image)
        edges_original = cv2.Canny(image, 100, 200)
        edges_cmap= plt.cm.gray(edges_original)
        saved_image_resized = saved_image.resize((400, 400))
        saved_image_tk = ImageTk.PhotoImage(saved_image_resized)
        
        # Convert processed images to Tkinter format
        image_list = [
            saved_image_tk,
            ImageTk.PhotoImage(Image.fromarray((image * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((blurred_image * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((segmented_image * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((cmap_image[:, :, :3] * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((edges_cmap[:, :, :3] * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((edges_sobel * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((edges_prewitt * 255).astype(np.uint8))),
            ImageTk.PhotoImage(Image.fromarray((edges_roberts * 255).astype(np.uint8))),
        ]
        titles_list = [
            'Initial Image',
            'Gray Image',
            'Gaussian Blurred',
            'Segmented Image',
            'Thresh Image',
            'Edge Detection',
            'Sobel Edges',
            'Prewitt Edges',
            'Roberts Edges',
        ]
        
        # Predict the tumor type using the image
        test_image_array = preprocess_image(image_path)
        predicted_class = predict_tumor_type(image_path)
        label_prediction.configure(text=f"The predicted class for the test image is:   {predicted_class}")

def update_image_and_title():
    global current_index
    if 0 <= current_index < len(image_list):
        my_label.configure(image=image_list[current_index])
        label_title.configure(text=titles_list[current_index])

def forward():
    global current_index
    if 0 <= current_index < len(image_list) - 1:
        current_index += 1
        update_image_and_title()
        update_navigation_buttons()

def back():
    global current_index
    if 0 < current_index < len(image_list):
        current_index -= 1
        update_image_and_title()
        update_navigation_buttons()

def update_navigation_buttons():
    global current_index
    # Update the navigation buttons based on the current index
    button_forward.configure(state=tk.NORMAL if current_index < len(image_list) - 1 else tk.DISABLED)
    button_back.configure(state=tk.NORMAL if current_index > 0 else tk.DISABLED)

# Create the Tkinter root window
root = CTk()
root.title("Brain Tumor Detection")
root.geometry("700x700")  # Adjusted size to fit new components
set_appearance_mode("dark")
# Create and place the button to open the image
btn_open = CTkButton(master=root, text="Open Image",corner_radius=32,
                        fg_color="transparent", hover_color="#4158D0",
                        border_width=2, command=open_image)
btn_open.grid(row=0, column=0, ipadx=20, ipady=5, padx=10, pady=10, sticky="W")

# Create a label to display the image
my_label = tk.Label(root)
my_label.grid(row=1, column=0, columnspan=3)

# Create a label to display the image title
label_title = CTkLabel(root,text=titles_list, font=("Helvetica", 14))
label_title.grid(row=2, column=0, columnspan=3)

# Create a label to display the file path
label_path = CTkLabel(root, text="No file selected")
label_path.grid(row=3, column=0, columnspan=3)

# Create a label to display the predicted tumor type
label_prediction = CTkLabel(root, text="The predicted class for the test image is: Not available", font=("Helvetica", 12))
label_prediction.grid(row=4, column=0, columnspan=3)

# Create navigation buttons
button_back = CTkButton(master=root, text="BACK", font=("Consolas", 12), corner_radius=32,
                        fg_color="transparent", hover_color="#4158D0",
                        border_width=2, command=back, state=tk.DISABLED)
button_exit = CTkButton(master=root, text="EXIT",corner_radius=32,
                        fg_color="transparent", hover_color="#4158D0",
                        border_width=2, command=root.quit)
button_forward = CTkButton(master=root, text="NEXT", font=("Consolas", 12),corner_radius=32,
                        fg_color="transparent", hover_color="#4158D0",
                        border_width=2, command=forward, state=tk.DISABLED)

# Create team label
label_team = CTkLabel(root, text="Developed by AceStars", font=("Helvetica", 14))
label_team.grid(row=6, column=1, pady=(10, 5))

button_back.grid(row=7, column=0, ipadx=20, ipady=5, padx=10, pady=10, sticky="W")
button_exit.grid(row=7, column=1, padx=10, pady=10)
button_forward.grid(row=7, column=2, ipadx=20, ipady=5, padx=10, pady=10, sticky="E")

# Start the Tkinter event loop
root.mainloop()
