import cv2
import os
import argparse

def images_to_video(image_folder, output_video, fps=17):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort()  
    
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created successfully: {output_video}")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--r', help='image folder path', default='../leftImg8bit/demoVideo/stuttgart_02', type=str) 

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    images_to_video(args.r, args.r+'.mp4')