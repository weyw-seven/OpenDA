'''
Grad-CAM 可视化
'''
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from MyNet import ResNet, BasicBlock

'''
加载和预处理图像
'''
def load_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

'''
可视化 Grad-CAM 结果
'''
def visualize_cam(cam_data, input_tensor, output_path=None):
    rgb_img = np.float32(input_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
    rgb_img = rgb_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    rgb_img = np.clip(rgb_img, 0, 1)
    cam_image = show_cam_on_image(rgb_img, cam_data, use_rgb=True)
    if output_path:
        plt.imsave(output_path, cam_image)
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()


def main():
    device = torch.device("cuda")
    model = ResNet(BasicBlock, [3, 3, 3], num_classes=100) 
    model.load_state_dict(torch.load('resnet_model.pth', map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.layer3[-1]  # 选择第三层最后一个残差块
    cam = GradCAM(model=model, target_layers=[target_layer])

    image_path = "imagenet_mini/val/n04081281/ILSVRC2012_val_00019466.JPEG"
    input_tensor = load_preprocess_image(image_path).to(device)

    # 使用 GradCAM
    with torch.enable_grad():
        cam_data = cam(input_tensor=input_tensor)
        cam_data = cam_data[0]

    # 可视化并保存结果
    visualize_cam(cam_data, input_tensor, output_path="8.jpg")


if __name__ == '__main__':
    main()
